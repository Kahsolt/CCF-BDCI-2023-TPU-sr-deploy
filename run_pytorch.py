import torch
from torch import Tensor
from torch.nn import Module

from run_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'>> device: {device}')

DEBUG_SHAPE = False
DEBUG_IMAGE = False


class TiledSRModel:

  def __init__(self, model_fp:Path, model_size:Tuple[int, int], padding=4):
    print(f'>> load model: {model_fp.stem}')
    self.model: Module = torch.load(model_fp, map_location='cpu')
    self.model = self.model.eval().to(device)
    print(f'>> param_cnt: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')
    try:    self.dtype = list(self.model.parameters())[0].dtype
    except: self.dtype = torch.float32
    self.upscale_rate = 4.0
    self.tile_size = model_size  # (h, w)
    self.padding = padding

  @property
  def tile_h(self): return self.tile_size[0]
  @property
  def tile_w(self): return self.tile_size[1]

  @torch.inference_mode()
  def __call__(self, im:ndarray, bs:int=4) -> ndarray:
    # [H, W, C=3]
    H, W, C = im.shape
    H_tgt, W_tgt = int(H * self.upscale_rate), int(W * self.upscale_rate)
    if DEBUG_SHAPE: print('im.shape:', im.shape)
    # tile count along aixs
    num_rows = math.ceil((H - self.padding) / (self.tile_h - self.padding))
    num_cols = math.ceil((W - self.padding) / (self.tile_w - self.padding))
    if DEBUG_SHAPE: print(f'tiles: {num_rows} x {num_cols}')
    # uncrop (zero padding)
    H_ex = num_rows * self.tile_h - ((num_rows - 1) * self.padding)
    W_ex = num_cols * self.tile_w - ((num_cols - 1) * self.padding)
    im_ex = np.zeros([H_ex, W_ex, C], dtype=im.dtype)
    if DEBUG_SHAPE: print('im_ex.shape:', im_ex.shape)
    # relocate top-left origin
    init_y = (H_ex - H) // 2
    init_x = (W_ex - W) // 2
    # paste original image in the center
    im_ex[init_y:init_y+H, init_x:init_x+W, :] = im

    if DEBUG_IMAGE: np_to_pil(im_ex).show()

    # [B=1, C=3, H_ex, W_ex]
    X = torch.from_numpy(np.transpose(im_ex, (2, 0, 1))).unsqueeze_(0)
    X = X.to(device=device, dtype=self.dtype)

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
    if DEBUG_SHAPE: print('n_tiles:', n_tiles)

    # forward & sew up tiles
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = torch.zeros([C, H_ex_tgt, W_ex_tgt], device=device, dtype=self.dtype)
    count  = torch.zeros([   H_ex_tgt, W_ex_tgt], device=device, dtype=torch.int32)
    while len(boxes_low):
      batch_low,  boxes_low  = boxes_low [:bs], boxes_low [bs:]
      batch_high, boxes_high = boxes_high[:bs], boxes_high[bs:]
      # [B, C, H_tile=192, W_tile=256]
      tiles_low = [X[:, :, slice_h, slice_w] for slice_h, slice_w in batch_low]
      if DEBUG_SHAPE: print('tile sizes:', [tuple(e.shape[2:]) for e in tiles_low])
      XT = torch.concat(tiles_low, dim=0)
      # [B, C, H_tile*F=764, W_tile*F=1024]
      tiles_high: List[Tensor] = self.model(XT)
      # paste to canvas
      for tile, (high_h, high_w) in zip(tiles_high, batch_high):
        count [   high_h, high_w] += 1
        canvas[:, high_h, high_w] += tile

    # handle overlap
    out_ex = torch.where(count > 1, canvas / count, canvas)

    if DEBUG_IMAGE: Image.fromarray((out_ex.permute([1, 2, 0]).cpu().numpy().clip(0.0, 1.0)*255).astype(np.uint8)).show()

    # crop
    fin_y = int(init_y * self.upscale_rate)
    fin_x = int(init_x * self.upscale_rate)
    out = out_ex[:, fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    # vrng, to HWC
    out = out.permute([1, 2, 0])
    # numpy & clip
    out_np: ndarray = out.cpu().numpy()
    out_np = out_np.clip(0.0, 1.0).astype(np.float32)

    if DEBUG_IMAGE: Image.fromarray((out_np*255).astype(np.uint8)).show()

    return out_np


def get_model(args):
  return TiledSRModel(args.model, args.model_size, padding=args.padding)


def process_images(args, model:Callable, paths:List[Path], niqe:List[float], runtime:List[float], result:List[dict]):
  total = len(paths)
  for idx, fp in enumerate(tqdm(paths)):
    # 加载图片
    img = Image.open(fp).convert('RGB')
    im_low = pil_to_np(img)

    # 模型推理
    start = time()
    im_high = model(im_low, bs=args.batch_size)
    end = time() - start
    runtime.append(end)

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
  args.backend = 'pytorch'
  args = process_args(args)

  run_eval(args, get_model, process_images)
