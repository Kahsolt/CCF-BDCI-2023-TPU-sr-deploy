from argparse import ArgumentParser

# TPU engine sdk
import sophon.sail as sail
sail.set_print_flag(False)
sail.set_dump_io_flag(False)

from utils import *


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/npuengine.py
class EngineOV:

  def __init__(self, model_path:str, device_id:int=0):
    if 'DEVICE_ID' in os.environ:
      device_id = int(os.environ['DEVICE_ID'])
      print('>> device_id is in os.environ. and device_id = ', device_id)
    try:
      self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
    except Exception as e:
      print('load model error; please check model path and device status')
      print('>> model_path: ', model_path)
      print('>> device_id: ', device_id)
      print('>> sail.Engine error: ', e)
      raise e

    self.model_path  = model_path
    self.device_id   = device_id
    self.graph_name  = self.model.get_graph_names()[0]
    self.input_name  = self.model.get_input_names(self.graph_name)
    self.output_name = self.model.get_output_names(self.graph_name)

  def __str__(self):
    return f'EngineOV: model_path={self.model_path}, device_id={self.device_id}'

  def __call__(self, values:list):
    assert isinstance(values, list), 'input should be a list'
    input = { self.input_name[i]: values[i] for i in range(len(values)) }
    output = self.model.process(self.graph_name, input)
    return [output[name] for name in self.output_name]


class TiledSRModel:

  def __init__(self, model_fp:Path, model_size:Tuple[int, int], padding=16, device_id=0):
    print(f'>> load model: {model_fp.stem}')
    self.model = EngineOV(str(model_fp), device_id)
    self.bs = 1
    self.upscale_rate = 4.0
    self.tile_size = model_size  # (h, w)
    self.padding = padding

  @property
  def tile_h(self): return self.tile_size[0]
  @property
  def tile_w(self): return self.tile_size[1]

  def __call__(self, im:ndarray) -> ndarray:
    # [H, W, C=3]
    H, W, C = im.shape
    H_tgt, W_tgt = int(H * self.upscale_rate), int(W * self.upscale_rate)
    # tile count along aixs
    num_rows = math.ceil((H - self.padding) / (self.tile_h - self.padding))
    num_cols = math.ceil((W - self.padding) / (self.tile_w - self.padding))
    # uncrop (zero padding)
    H_ex = num_rows * self.tile_h - ((num_rows - 1) * self.padding)
    W_ex = num_cols * self.tile_w - ((num_cols - 1) * self.padding)
    im_ex = np.zeros([H_ex, W_ex, C], dtype=im.dtype)
    # relocate top-left origin
    init_y = (H_ex - H) // 2
    init_x = (W_ex - W) // 2
    # paste original image in the center
    im_ex[init_y:init_y+H, init_x:init_x+W, :] = im

    # [B=1, C=3, H_ex, W_ex]
    X = np.expand_dims(np.transpose(im_ex, (2, 0, 1)), axis=0)

    # break up tiles
    boxes_low:  List[Box] = []
    boxes_high: List[Box] = []
    y = 0
    while y + self.padding < H_ex:
      x = 0
      while x + self.padding < W_ex:
        boxes_low.append((
          slice(y, y + self.tile_h), 
          slice(x, x + self.tile_w),
        ))
        boxes_high.append((
          slice(int(y * self.upscale_rate), int((y + self.tile_h) * self.upscale_rate)), 
          slice(int(x * self.upscale_rate), int((x + self.tile_w) * self.upscale_rate)),
        ))
        x += self.tile_w - self.padding
      y += self.tile_h - self.padding
    n_tiles = len(boxes_low)
    assert n_tiles == num_rows * num_cols

    # forward & sew up tiles
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = np.zeros([C, H_ex_tgt, W_ex_tgt], dtype=X.dtype)
    count  = np.zeros([   H_ex_tgt, W_ex_tgt], dtype=np.int32)
    while len(boxes_low):
      batch_low,  boxes_low  = boxes_low [:self.bs], boxes_low [self.bs:]
      batch_high, boxes_high = boxes_high[:self.bs], boxes_high[self.bs:]
      # [B, C, H_tile=192, W_tile=256]
      tiles_low = [X[:, :, slice_h, slice_w] for slice_h, slice_w in batch_low]
      XT = np.concatenate(tiles_low, axis=0)
      # [B, C, H_tile*F=764, W_tile*F=1024]
      tiles_high: List[ndarray] = self.model([XT])[0]
      # paste to canvas
      for tile, (high_h, high_w) in zip(tiles_high, batch_high):
        count [   high_h, high_w] += 1
        canvas[:, high_h, high_w] += tile

    # handle overlap
    out_ex = np.where(count > 1, canvas / count, canvas)
    # crop
    fin_y = int(init_y * self.upscale_rate)
    fin_x = int(init_x * self.upscale_rate)
    out = out_ex[:, fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    # vrng, to HWC
    out = np.transpose(out, [1, 2, 0])
    # numpy & clip
    out = out.clip(0.0, 1.0).astype(np.float32)
    return out


def run(args):
  # in/out paths
  if Path(args.input).is_file():
    paths = [Path(args.input)]
  else:
    paths = [Path(fp) for fp in sorted(glob.glob(os.path.join(str(args.input), '*')))]
  if args.limit > 0: paths = paths[:args.limit]
  if args.save: Path(args.output).mkdir(parents=True, exist_ok=True)

  # setup model
  model = TiledSRModel(args.model, args.model_size, padding=args.padding, device_id=args.device)

  # workers & task
  start_all = time()
  total = len(paths)
  result:  List[dict]  = []
  runtime: List[float] = []
  niqe:    List[float] = []
  for idx, fp in enumerate(tqdm(paths)):
    # 加载图片
    img = Image.open(fp).convert('RGB')
    im_low = np.asarray(img, dtype=np.float32) / 255.0

    # 模型推理
    start = time()
    im_high = model(im_low)
    end = time() - start
    runtime.append(end)

    # 保存图片
    if args.save:
      img = (np.asarray(im_high) * 255).astype(np.uint8)
      Image.fromarray(img).save(Path(args.output) / fp.name)

    # 计算niqe
    niqe_output = get_niqe(im_high)
    niqe.append(niqe_output)

    result.append({'img_name': fp.stem, 'runtime': format(end, '.4f'), 'niqe': format(niqe_output, '.4f')})

    if (idx + 1) % 10 == 0:
      print(f'>> [{idx+1}/{total}]: niqe {mean(niqe)}, time {mean(runtime)}')

  end_all = time()
  time_all = end_all - start_all
  runtime_avg = mean(runtime)
  niqe_avg = mean(niqe)
  print('time_all:',    time_all)
  print('runtime_avg:', runtime_avg)
  print('niqe_avg:',    niqe_avg)
  print('>> score:',    get_score(runtime_avg, niqe_avg))

  # gather results
  metrics = {
    'A': [{
      'model_size': os.path.getsize(args.model), 
      'time_all': time_all, 
      'runtime_avg': format(mean(runtime), '.4f'),
      'niqe_avg': format(mean(niqe), '.4f'), 
      'images': result,
    }]
  }
  print(f'>> saving to {args.report}')
  with open(args.report, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False)


def get_parser():
  parser = ArgumentParser()
  parser.add_argument('-D', '--device', type=int,  default=0,           help='TPU device id')
  parser.add_argument('-M', '--model',  type=Path, default='r-esrgan',  help='path to *.bmodel model ckpt, , or folder name under path models/')
  parser.add_argument('--model_size',   type=str,                       help='model input size like 200 or 196,256')
  parser.add_argument('--tile_size',    type=int,  default=196)
  parser.add_argument('--padding',      type=int,  default=16)
  parser.add_argument('-I', '--input',  type=Path, default=IN_PATH,     help='input image or folder')
  parser.add_argument('-L', '--limit',  type=int,  default=-1,          help='limit run sample count')
  parser.add_argument('--save',         action='store_true',            help='save sr images')
  return parser


def get_args(parser:ArgumentParser=None):
  parser = parser or get_parser()
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()

  fp = Path(args.model)
  if not fp.is_file():
    dp: Path = MODEL_PATH / args.model
    assert dp.is_dir(), 'should be a folder name under path models/'
    fps = [fp for fp in dp.iterdir() if fp.suffix == '.bmodel']
    assert len(fps) == 1, 'folder contains mutiplt *.bmodel files'
    args.model = fps[0]

  args.model_size = fix_model_size(args.model_size)

  args.log_dp = OUT_PATH / Path(args.model).stem
  args.log_dp.mkdir(exist_ok=True)
  args.output = args.log_dp / 'test_sr'
  args.report = args.log_dp / 'test.json'

  run(args)
