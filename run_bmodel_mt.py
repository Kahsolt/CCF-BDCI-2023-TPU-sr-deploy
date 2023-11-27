from threading import Thread

from run_bmodel import *


class TiledSRBModelMT(TiledSR):

  def __init__(self, model_fp:Path, model_size:Tuple[int, int], padding:int=4, n_workers:int=4, device_id:int=0):
    super().__init__(model_size, padding)

    print(f'>> n_workers: {n_workers}')
    self.n_workers = n_workers
    print(f'>> load model: {model_fp.stem}')
    self.models = [EngineOV(str(model_fp), device_id) for _ in range(n_workers)]
    self.bs = self.models[0].input_shape[0]
    assert self.bs == 1, 'only support batch_size == 1'

  def forward_tiles(self, X:ndarray, boxes_low:List[Box], boxes_high:List[Box]) -> ndarray:
    def task_tile_jobs(X:ndarray, idxs:range, boxes_low:List[Box], boxes_high:List[ndarray]):
      n_tiles = len(boxes_low)
      for i in idxs:
        if i >= n_tiles: break
        low_h,  low_w  = boxes_low [i] 
        high_h, high_w = boxes_high[i]

        # [C, H_tile=192, W_tile=256]
        tile_low = X[:, low_h, low_w].copy()      # NOTE: will produce bad if not copy
        # [C, H_tile*F=768, W_tile*F=1024]
        XT = np.expand_dims(tile_low, axis=0)
        YT = self.models[i % self.n_workers](XT)
        tile_high = YT[0]
        # [C, H_tile*F=768, W_tile*F=1024]
        canvas[:, high_h, high_w] = tile_high

    C, H_ex , W_ex = X.shape
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = np.zeros([C, H_ex_tgt, W_ex_tgt], dtype=X.dtype)
    n_tiles = len(boxes_low)
    n_jobs = math.ceil(n_tiles / self.n_workers)

    thrs = [Thread(target=task_tile_jobs, args=(X, range(i*n_jobs, (i+1)*n_jobs), boxes_low, boxes_high)) for i in range(self.n_workers)]
    for thr in thrs: thr.start()
    for thr in thrs: thr.join()
    return canvas


class TiledSRBModelTileMT(TiledSRBModelMT):

  ''' simple non-overlaping tiling '''

  def __call__(self, im:ndarray) -> ndarray:
    # R
    R = self.upscale_rate
    # [H, W, C=3]
    H, W, C = im.shape
    H_tgt, W_tgt = int(H * R), int(W * R)
    # tile count along aixs
    num_rows = math.ceil(H / self.tile_h)
    num_cols = math.ceil(W / self.tile_w)
    # uncrop (zero padding)
    H_ex = num_rows * self.tile_h
    W_ex = num_cols * self.tile_w
    # pad to expanded canvas
    d_H = H_ex - H ; d_H_2 = d_H // 2
    d_W = W_ex - W ; d_W_2 = d_W // 2
    ts_pad = time()
    im_ex = np.pad(im, ((d_H_2, d_H-d_H_2), (d_W_2, d_W-d_W_2), (0, 0)), mode='constant', constant_values=0.0)
    if DEBUG_TIME: print('ts_pad:', time() - ts_pad)

    # [C=3, H_ex, W_ex]
    X = np.transpose(im_ex, (2, 0, 1))

    # break up tiles
    boxes_low:  List[Box] = []
    boxes_high: List[Box] = []
    y = 0
    while y < H_ex:
      x = 0
      while x < W_ex:
        boxes_low.append((
          slice(y, y + self.tile_h), 
          slice(x, x + self.tile_w),
        ))
        boxes_high.append((
          slice(int(y * R), int((y + self.tile_h) * R)), 
          slice(int(x * R), int((x + self.tile_w) * R)),
        ))
        x += self.tile_w
      y += self.tile_h

    # forward & sew up tiles
    canvas = self.forward_tiles(X, boxes_low, boxes_high)

    # relocate top-left origin
    fin_y = int((H_ex - H) // 2 * R)
    fin_x = int((W_ex - W) // 2 * R)
    # crop
    out = canvas[:, fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    # to HWC
    return np.transpose(out, [1, 2, 0])


def get_model(args):
  return TiledSRBModelTileMT(args.model, args.model_size, args.padding, args.n_workers, device_id=args.device)


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--n_workers', default=4, type=int)
  args = get_args(parser)
  args.backend = 'bmodel'
  args = process_args(args)

  assert args.padding == 0, 'only support padding == 0'
  assert args.batch_size == 1, 'only support batch_size == 1'

  run_eval(args, get_model, process_images)
