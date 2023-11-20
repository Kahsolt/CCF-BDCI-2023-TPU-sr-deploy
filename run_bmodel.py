# TPU engine sdk
import sophon.sail as sail
sail.set_print_flag(False)
sail.set_dump_io_flag(False)

from run_utils import *


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
    for i in range(len(boxes_low)):
      low_slices  = boxes_low [i]
      high_slices = boxes_high[i]
      # [B=1, C, H_tile=192, W_tile=256]
      low_h, low_w = low_slices
      XT = X[:, :, low_h, low_w]
      # [B=1, C, H_tile*F=764, W_tile*F=1024]
      YT: ndarray = self.model([XT])[0][0]
      # paste to canvas
      high_h, high_w = high_slices
      count [   high_h, high_w] += 1
      canvas[:, high_h, high_w] += YT

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


def get_model(args):
  return TiledSRModel(args.model, args.model_size, padding=args.padding, device_id=args.device)


def process_images(args, model:Callable, paths:List[Path], niqe:List[float], runtime:List[float], result:List[dict]):
  total = len(paths)
  for idx, fp in enumerate(tqdm(paths)):
    # 加载图片
    img = Image.open(fp).convert('RGB')
    im_low = pil_to_np(img)

    # 模型推理
    start = time()
    im_high: ndarray = model(im_low)
    end = time() - start
    runtime.append(end)

    im_high = im_high.clip(0.0, 1.0)    # vrng 0~1
    img_high = None
    
    # 后处理
    if args.postprocess:
      img_high = img_high or np_to_pil(im_high)
      img_high = img_high.filter(ImageFilter.DETAIL)
      im_high = pil_to_np(img_high)

    # 保存图片
    if args.save:
      img_high = img_high or np_to_pil(im_high)
      img_high.save(Path(args.output) / fp.name)

    # 计算niqe
    niqe_output = get_niqe(im_high)
    niqe.append(niqe_output)

    result.append({'img_name': fp.stem, 'runtime': format(end, '.4f'), 'niqe': format(niqe_output, '.4f')})

    if (idx + 1) % 10 == 0:
      print(f'>> [{idx+1}/{total}]: niqe {mean(niqe)}, time {mean(runtime)}')


if __name__ == '__main__':
  args = get_args()
  args.backend = 'bmodel'
  args.batch_size = 1
  args = process_args(args)

  run_eval(args, get_model, process_images)
