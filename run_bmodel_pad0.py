from run_bmodel import *

DEBUG_TIME = True


class TiledSRModelNoPad(TiledSRModel):

  def __init__(self, model_fp:Path, model_size:Tuple[int, int], device_id=0):
    super().__init__(model_fp, model_size, 0, device_id)

  def __call__(self, im:ndarray) -> ndarray:
    if DEBUG_TIME: ts_cvs = time()
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
    if DEBUG_TIME: print('ts_cvs:', time() - ts_cvs)

    # break up tiles
    if DEBUG_TIME: ts_box = time()
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
    #assert len(boxes_low) == num_rows * num_cols
    if DEBUG_TIME: print('ts_box:', time() - ts_box)

    # forward & sew up tiles
    if DEBUG_TIME: ts_tiles = time()
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = np.zeros([C, H_ex_tgt, W_ex_tgt], dtype=X.dtype)
    for i in range(len(boxes_low)):
      low_slices  = boxes_low [i]
      high_slices = boxes_high[i]
      # [B=1, C, H_tile=192, W_tile=256]
      low_h, low_w = low_slices
      XT = X[:, :, low_h, low_w]
      # [B=1, C, H_tile*F=764, W_tile*F=1024]
      if DEBUG_TIME: ts_tile = time()
      YT: ndarray = self.model([XT])[0][0]
      if DEBUG_TIME: print('ts_tile:', time() - ts_tile)
      # paste to canvas
      high_h, high_w = high_slices
      canvas[:, high_h, high_w] = YT
    if DEBUG_TIME: print('ts_tiles:', time() - ts_tiles)

    # crop
    if DEBUG_TIME: ts_pp = time()
    fin_y = int(init_y * self.upscale_rate)
    fin_x = int(init_x * self.upscale_rate)
    out = canvas[:, fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    # to HWC
    out = np.transpose(out, [1, 2, 0])
    if DEBUG_TIME:
      ts_end = time()
      print('ts pp:', ts_end - ts_pp)
      print('ts all:', ts_end - ts_cvs)
    return out


def get_model(args):
  return TiledSRModelNoPad(args.model, args.model_size, device_id=args.device)


if __name__ == '__main__':
  args = get_args()
  args.backend = 'bmodel'
  args.batch_size = 1
  args = process_args(args)

  run_eval(args, get_model, process_images)
