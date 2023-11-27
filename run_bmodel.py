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
      print('>> device_id is in os.environ. use device_id = ', device_id)
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
    self.input_name  = self.model.get_input_names (self.graph_name)[0]
    self.output_name = self.model.get_output_names(self.graph_name)[0]

    self.input_shape = self.model.get_input_shape(self.graph_name, self.input_name)
    self.input_dtype = self.model.get_input_dtype(self.graph_name, self.input_name)
    print('>> input_shape:', self.input_shape)
    print('>> input_dtype:', self.input_dtype)
    self.output_shape = self.model.get_output_shape(self.graph_name, self.output_name)
    self.output_dtype = self.model.get_output_dtype(self.graph_name, self.output_name)
    print('>> output_shape:', self.output_shape)
    print('>> output_dtype:', self.output_dtype)

  def __str__(self):
    return f'EngineOV: model_path={self.model_path}, device_id={self.device_id}'

  def __call__(self, value:ndarray) -> ndarray:
    input = { self.input_name: value }
    output = self.model.process(self.graph_name, input)
    return output[self.output_name]


class TiledSRBModel(TiledSR):

  def __init__(self, model_fp:Path, model_size:Tuple[int, int], padding:int=4, device_id:int=0):
    super().__init__(model_size, padding)

    print(f'>> load model: {model_fp.stem}')
    self.model = EngineOV(str(model_fp), device_id)
    self.bs = self.model.input_shape[0]

  def forward_tiles(self, X:ndarray, boxes_low:List[Box], boxes_high:List[Box], P_ex:int=None, ret_cnt:bool=False) -> ndarray:
    C, H_ex , W_ex = X.shape

    if DEBUG_TIME: ts_tiles = time()
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = np.zeros([C, H_ex_tgt, W_ex_tgt], dtype=X.dtype)
    if ret_cnt:
      count = np.zeros([H_ex_tgt, W_ex_tgt], dtype=np.int32)
    while len(boxes_low):
      batch_low,  boxes_low  = boxes_low [:self.bs], boxes_low [self.bs:]
      batch_high, boxes_high = boxes_high[:self.bs], boxes_high[self.bs:]
      # [B, C, H_tile=192, W_tile=256]
      tiles_low = [X[:, slice_h, slice_w] for slice_h, slice_w in batch_low]
      # batch count pad
      while len(tiles_low) < self.bs:
        tiles_low.append(np.zeros_like(tiles_low[-1]))
      XT = np.stack(tiles_low, axis=0)
      # [B, C, H_tile*F=764, W_tile*F=1024]
      if DEBUG_TIME: ts_tile = time()
      tiles_high: ndarray = self.model(XT)
      if DEBUG_TIME: print('ts_tile:', time() - ts_tile)
      # trim batch count pad
      tiles_high = tiles_high[:len(batch_low)]
      # paste to canvas
      for tile, (high_h, high_w) in zip(tiles_high, batch_high):
        if ret_cnt:
          count [high_h, high_w] += 1
        if P_ex is None:
          canvas[:, high_h, high_w] += tile
        else:
          canvas[:, high_h, high_w] += tile[:, P_ex:-P_ex, P_ex:-P_ex]
    if DEBUG_TIME: print('ts_tiles:', time() - ts_tiles)

    return (canvas, count) if ret_cnt else canvas


class TiledSRBModelTileMixin:

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

class TiledSRBModelTile(TiledSRBModel, TiledSRBModelTileMixin):

  ''' simple non-overlaping tiling '''

  pass


class TiledSRBModelOverlap(TiledSRBModel):

  ''' overlapped tiling with simple counting average '''

  def __call__(self, im:ndarray) -> ndarray:
    if DEBUG_TIME: ts_cvs = time()
    # P & R
    P = self.padding
    R = self.upscale_rate
    # [H, W, C=3]
    H, W, C = im.shape
    H_tgt, W_tgt = int(H * R), int(W * R)
    # tile count along aixs
    num_rows = math.ceil((H - P) / (self.tile_h - P))
    num_cols = math.ceil((W - P) / (self.tile_w - P))
    # uncrop (zero padding)
    H_ex = num_rows * self.tile_h - ((num_rows - 1) * P)
    W_ex = num_cols * self.tile_w - ((num_cols - 1) * P)
    im_ex = np.zeros([H_ex, W_ex, C], dtype=im.dtype)
    # relocate top-left origin
    init_y = (H_ex - H) // 2
    init_x = (W_ex - W) // 2
    # paste original image in the center
    im_ex[init_y:init_y+H, init_x:init_x+W, :] = im

    # [B=1, C=3, H_ex, W_ex]
    X = np.transpose(im_ex, (2, 0, 1))
    if DEBUG_TIME: print('ts_cvs:', time() - ts_cvs)

    # break up tiles
    if DEBUG_TIME: ts_box = time()
    boxes_low:  List[Box] = []
    boxes_high: List[Box] = []
    y = 0
    while y + P < H_ex:
      x = 0
      while x + P < W_ex:
        boxes_low.append((
          slice(y, y + self.tile_h), 
          slice(x, x + self.tile_w),
        ))
        boxes_high.append((
          slice(int(y * R), int((y + self.tile_h) * R)), 
          slice(int(x * R), int((x + self.tile_w) * R)),
        ))
        x += self.tile_w - P
      y += self.tile_h - P
    #assert len(boxes_low) == num_rows * num_cols
    if DEBUG_TIME: print('ts_box:', time() - ts_box)

    # forward & sew up tiles
    canvas, count = self.forward_tiles(X, boxes_low, boxes_high, ret_cnt=True)

    # crop
    if DEBUG_TIME: ts_crop = time()
    fin_y = int(init_y * R)
    fin_x = int(init_x * R)
    cvs_crop = canvas[:, fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    cnt_crop = count [   fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    # handle overlap
    out = np.where(cnt_crop > 1, cvs_crop / cnt_crop, cvs_crop)
    # to HWC
    out = np.transpose(out, [1, 2, 0])
    if DEBUG_TIME:
      ts_end = time()
      print('ts crop:', ts_end - ts_crop)
      print('ts __call__:', ts_end - ts_cvs)
    return out


class TiledSRBModelCrop(TiledSRBModel):

  ''' non-overlaping tiling with margin cropping '''

  @property
  def tile_h(self): return self.model_size[0] - self.padding * 2
  @property
  def tile_w(self): return self.model_size[1] - self.padding * 2

  def __call__(self, im:ndarray) -> ndarray:
    if DEBUG_TIME: ts_cvs = time()
    # P & R
    P = self.padding
    R = self.upscale_rate
    P_ex = int(P * R)
    # [H, W, C=3]
    H, W, C = im.shape
    H_tgt, W_tgt = int(H * R), int(W * R)
    # tile count along aixs
    num_rows = math.ceil(H / self.tile_h)
    num_cols = math.ceil(W / self.tile_w)
    # uncrop (zero padding)
    H_ex = num_rows * self.tile_h + P * 2
    W_ex = num_cols * self.tile_w + P * 2
    im_ex = np.zeros([H_ex, W_ex, C], dtype=im.dtype)
    # relocate top-left origin
    init_y = (H_ex - H) // 2 + P
    init_x = (W_ex - W) // 2 + P
    # paste original image in the center
    im_ex[init_y:init_y+H, init_x:init_x+W, :] = im

    # [C=3, H_ex, W_ex]
    X = np.transpose(im_ex, (2, 0, 1))
    if DEBUG_TIME: print('ts_cvs:', time() - ts_cvs)

    # break up tiles
    if DEBUG_TIME: ts_box = time()
    boxes_low:  List[Box] = []
    boxes_high: List[Box] = []
    y = P
    while y + P < H_ex:
      x = P
      while x + P < W_ex:
        boxes_low.append((
          # expand to model_size
          slice(y - P, y + self.tile_h + P), 
          slice(x - P, x + self.tile_w + P),
        ))
        boxes_high.append((
          # keep cropped to tile_size * upscale_rate
          slice(int(y * R), int((y + self.tile_h) * R)), 
          slice(int(x * R), int((x + self.tile_w) * R)),
        ))
        x += self.tile_w
      y += self.tile_h
    #assert len(boxes_low) == num_rows * num_cols
    if DEBUG_TIME: print('ts_box:', time() - ts_box)

    # forward & sew up tiles
    canvas = self.forward_tiles(X, boxes_low, boxes_high, P_ex=P_ex)

    # crop
    if DEBUG_TIME: ts_crop = time()
    fin_y = int(init_y * R)
    fin_x = int(init_x * R)
    out = canvas[:, fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    # to HWC
    out = np.transpose(out, [1, 2, 0])
    if DEBUG_TIME:
      ts_end = time()
      print('ts crop:', ts_end - ts_crop)
      print('ts __call__:', ts_end - ts_cvs)
    return out


def get_model_tile(args):
  return TiledSRBModelTile(args.model, args.model_size, args.padding, device_id=args.device)


def get_model_overlap(args):
  return TiledSRBModelOverlap(args.model, args.model_size, args.padding, device_id=args.device)


def get_model_crop(args):
  # NOTE: negativate args.padding here
  return TiledSRBModelCrop(args.model, args.model_size, -args.padding, device_id=args.device)


def get_model(args):
  if   args.padding == 0: return get_model_tile(args)
  elif args.padding  > 0: return get_model_overlap(args)
  else:                   return get_model_crop(args)


if __name__ == '__main__':
  args = get_args()
  args.backend = 'bmodel'
  args = process_args(args)

  run_eval(args, get_model, process_images)
