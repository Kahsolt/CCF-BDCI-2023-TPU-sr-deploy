from threading import Thread

from run_bmodel import *


class TiledSRBModelTileMT(TiledSR, TiledSRBModelTileMixin):

  ''' simple non-overlaping tiling, multi-thread '''

  def __init__(self, model_fp:Path, model_size:Tuple[int, int], padding:int=4, n_workers:int=4, device_id:int=0):
    super().__init__(model_size, padding)

    print(f'>> n_workers: {n_workers}')
    self.n_workers = n_workers
    print(f'>> load model: {model_fp.stem}')
    self.model = EngineOV(str(model_fp), device_id)
    self.bs = self.model.input_shape[0]
    assert self.bs == 1, 'only support batch_size == 1'

  def forward_tiles(self, X:ndarray, boxes_low:List[Box], boxes_high:List[Box]) -> ndarray:
    n_tiles = len(boxes_low)
    n_jobs = math.ceil(n_tiles / self.n_workers)
  
    ''' split '''
    def task_split_tiles(X:ndarray, idxs:range, boxes_low:List[Box], tiles_low:List[ndarray]):
      n_tiles = len(boxes_low)
      for i in idxs:
        if i >= n_tiles: break

        low_h, low_w = boxes_low[i] 
        tile_low = X[:, low_h, low_w].copy()      # NOTE: will produce bad if not copy
        tiles_low[i] = np.expand_dims(tile_low, axis=0)

    if DEBUG_TIME: ts_split_tiles = time()
    tiles_low = [None] * n_tiles
    thrs = [Thread(target=task_split_tiles, args=(X, range(i*n_jobs, (i+1)*n_jobs), boxes_low, tiles_low)) for i in range(self.n_workers)]
    for thr in thrs: thr.start()
    for thr in thrs: thr.join()
    if DEBUG_TIME: print('>> ts_split_tiles:', time() - ts_split_tiles)

    ''' upscale '''
    if DEBUG_TIME: ts_upscale_tiles = time()
    tiles_high = [self.model(tile_low)[0] for tile_low in tiles_low]

    if DEBUG_TIME: print('>> ts_upscale_tiles:', time() - ts_upscale_tiles)

    ''' combine '''
    if DEBUG_TIME: ts_combine_tiles = time()
    def task_combine_tiles(canvas:ndarray, idxs:range, boxes_high:List[Box], tiles_high:List[ndarray]):
      n_tiles = len(boxes_high)
      for i in idxs:
        if i >= n_tiles: break

        high_h, high_w = boxes_high[i]
        canvas[:, high_h, high_w] = tiles_high[i]

    C, H_ex , W_ex = X.shape
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = np.empty([C, H_ex_tgt, W_ex_tgt], dtype=X.dtype)

    thrs = [Thread(target=task_combine_tiles, args=(canvas, range(i*n_jobs, (i+1)*n_jobs), boxes_high, tiles_high)) for i in range(self.n_workers)]
    for thr in thrs: thr.start()
    for thr in thrs: thr.join()
    if DEBUG_TIME: print('>> ts_combine_tiles:', time() - ts_combine_tiles)

    return canvas


class TiledSRBModelTileMTME(TiledSR, TiledSRBModelTileMixin):

  ''' simple non-overlaping tiling, multi-thread & multi-engine'''

  def __init__(self, model_fp:Path, model_size:Tuple[int, int], padding:int=4, n_workers:int=4, device_id:int=0):
    super().__init__(model_size, padding)

    print(f'>> n_workers: {n_workers}')
    self.n_workers = n_workers
    print(f'>> load model: {model_fp.stem}')
    self.models = [EngineOV(str(model_fp), device_id) for _ in range(n_workers)]
    self.bs = self.models[0].input_shape[0]
    assert self.bs == 1, 'only support batch_size == 1'

  def forward_tiles(self, X:ndarray, boxes_low:List[Box], boxes_high:List[Box]) -> ndarray:
    def task(tid:int, X:ndarray, canvas:ndarray, idxs:range, boxes_low:List[Box], boxes_high:List[Box]):
      n_tiles = len(boxes_low)
      for i in idxs:
        if i >= n_tiles: break
        low_h,  low_w  = boxes_low[i]
        high_h, high_w = boxes_high[i]
        
        tile_low = X[:, low_h, low_w].copy()      # NOTE: will produce bad if not copy
        tile_low = np.expand_dims(tile_low, axis=0)
        tile_high = self.models[tid](tile_low)[0]
        canvas[:, high_h, high_w] = tile_high

    C, H_ex , W_ex = X.shape
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = np.empty([C, H_ex_tgt, W_ex_tgt], dtype=X.dtype)
    n_jobs = math.ceil(len(boxes_low) / self.n_workers)

    thrs = [Thread(target=task, args=(i, X, canvas, range(i*n_jobs, (i+1)*n_jobs), boxes_low, boxes_high)) for i in range(self.n_workers)]
    for thr in thrs: thr.start()
    for thr in thrs: thr.join()

    return canvas


def get_model(args):
  if args.n_engines == 'single':
    return TiledSRBModelTileMT(args.model, args.model_size, args.padding, args.n_workers, device_id=args.device)
  elif args.n_engines == 'multi':
    return TiledSRBModelTileMTME(args.model, args.model_size, args.padding, args.n_workers, device_id=args.device)


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--n_workers', default=4, type=int)
  parser.add_argument('--n_engines', default='multi', choices=['single', 'multi'])
  args = get_args(parser)
  args.backend = 'bmodel'
  args = process_args(args)

  if args.padding != 0:
    print('>> only support padding == 0')
    args.padding = 0
  if args.batch_size != 1:
    print('>> only support batch_size == 1')
    args.batch_size = 1

  run_eval(args, get_model, process_images)
