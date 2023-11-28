#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/17 

import warnings ; warnings.simplefilter('ignore', category=RuntimeWarning)

import os
import sys
import math
import json
from time import time
from pathlib import Path
from threading import Thread
from argparse import ArgumentParser
from pprint import pprint as pp
from typing import Tuple, List

from tqdm import tqdm
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
import numpy as np
from numpy import ndarray

BASE_PATH  = Path(__file__).absolute().parent.parent
LIB_PATH   = BASE_PATH / 'metrics'
IN_PATH    = BASE_PATH / 'dataset'
OUT_PATH   = BASE_PATH / 'results'
MODEL_PATH = BASE_PATH / 'models'

# TPU engine sdk
import sophon.sail as sail
sail.set_print_flag(False)
sail.set_dump_io_flag(False)
# contest scaffold
sys.path.append(str(BASE_PATH))
sys.path.append(str(LIB_PATH))
from metrics.niqe import calculate_niqe     # use the offical scorer, very slow though :(

Box = Tuple[slice, slice]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0

def get_score(niqe_avg:float, runtime_avg:float) -> float:
  try: return math.sqrt(7 - niqe_avg) / runtime_avg * 200
  except: return -1

def pil_to_np(img:PILImage) -> ndarray:
  return np.asarray(img, dtype=np.float32) / 255.0

def np_to_pil(im:ndarray) -> PILImage:
  return Image.fromarray(np.asarray(im.clip(0.0, 1.0) * 255).astype(np.uint8))


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/npuengine.py
class EngineOV:

  def __init__(self, model_path:str, device_id:int=0, tid:int=0):
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

    self.input_shape  = self.model.get_input_shape (self.graph_name, self.input_name)
    self.input_dtype  = self.model.get_input_dtype (self.graph_name, self.input_name)
    self.output_shape = self.model.get_output_shape(self.graph_name, self.output_name)
    self.output_dtype = self.model.get_output_dtype(self.graph_name, self.output_name)
    if tid == 0:
      print('>> input_shape:', self.input_shape)
      print('>> input_dtype:', self.input_dtype)
      print('>> output_shape:', self.output_shape)
      print('>> output_dtype:', self.output_dtype)

  def __str__(self):
    return f'EngineOV: model_path={self.model_path}, device_id={self.device_id}'

  def __call__(self, value:ndarray) -> ndarray:
    input = { self.input_name: value }
    output = self.model.process(self.graph_name, input)
    return output[self.output_name]


class TiledSRBModelTileMTME:

  ''' simple non-overlaping tiling, multi-thread & multi-engine'''

  def __init__(self, model_fp:Path, n_workers:int=4, device_id:int=0):
    print('>> tiler:', self.__class__.__name__)
    self.upscale_rate = 4.0
    print(f'>> n_workers: {n_workers}')
    self.n_workers = n_workers
    print(f'>> load model: {model_fp.stem}')
    self.models = [EngineOV(str(model_fp), device_id, tid) for tid in range(n_workers)]
    B, C, H, W = self.models[0].input_shape
    assert B == 1, 'only support batch_size == 1'
    self.model_size = (H, W)

  @property
  def tile_h(self): return self.model_size[0]
  @property
  def tile_w(self): return self.model_size[1]

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
    im_ex = np.pad(im, ((d_H_2, d_H-d_H_2), (d_W_2, d_W-d_W_2), (0, 0)), mode='constant', constant_values=0.0)

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


def run(args):
  fps = sorted(Path(args.input).iterdir())
  n_images = len(fps)
  print(f'>> evaulating {n_images} images')

  model = TiledSRBModelTileMTME(args.model, args.n_workers, device_id=args.device)

  niqe:    List[float] = []
  runtime: List[float] = []
  result:  List[dict]  = []
  start_all = time()

  for idx, fp in enumerate(tqdm(fps)):
    img = Image.open(fp).convert('RGB')
    im_low = pil_to_np(img)

    start = time()
    im_high: ndarray = model(im_low)
    end = time() - start
    runtime.append(round(end, 4))

    img_high = np_to_pil(im_high)
    if args.postprocess:
      img_high = img_high.filter(ImageFilter.EDGE_ENHANCE)
    fp_save = Path(args.output) / fp.name
    img_high.save(fp_save)

    img = Image.open(fp_save).convert('RGB')
    im_high = pil_to_np(img)[:, :, ::-1]    # RGB2BGR

    niqe_output = calculate_niqe(im_high, 0, input_order='HWC', convert_to='y')
    niqe.append(round(niqe_output, 4))

    result.append({'img_name': fp.stem, 'runtime': format(end, '.4f'), 'niqe': format(niqe_output, '.4f')})

    if (idx + 1) % 10 == 0:
      running_niqe = mean(niqe)
      running_time = mean(runtime)
      running_score = get_score(running_niqe, running_time)
      print(f'>> [{idx+1}/{n_images}]: niqe {running_niqe}, time {running_time}, score {running_score}')

  time_all = time() - start_all
  runtime_avg = mean(runtime)
  niqe_avg = mean(niqe)
  print('time_all:', time_all)
  print('runtime_avg:', runtime_avg)
  print('niqe_avg:', niqe_avg)
  print('>> score:', get_score(niqe_avg, runtime_avg))

  ranklist = 'A' if args.dataset == 'test' else 'B'
  metrics = {
    ranklist: [{
      'model_size': os.path.getsize(str(args.model)), 
      'time_all': time_all, 
      'runtime_avg': format(runtime_avg, '.4f'),
      'niqe_avg': format(niqe_avg, '.4f'), 
      'images': result,
    }]
  }
  print(f'>> saving to {args.report}')
  with open(args.report, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model',   type=Path, default='espcn_ex', help=f'path to *.bmodel file, or filename under {MODEL_PATH!r}')
  parser.add_argument('-D', '--dataset', type=str,  default='val', choices=['test', 'val'])
  parser.add_argument('--device',    default=0, type=int, help='TPU device id')
  parser.add_argument('--n_workers', default=4, type=int, help='multi-threading')
  args, _ = parser.parse_known_args()

  fp = Path(args.model)
  if not fp.is_file():
    dp: Path = MODEL_PATH / fp.with_suffix('.bmodel')
    args.model = dp
  assert args.model.is_file(), f'>> model path {args.model} is not a file :('
  if args.model.stem.endswith('_ex'):   # espcn_ex model requires CPU postprocess
    args.postprocess = True
  else:                                 # espcn_um already has embedded postprocess
    args.postprocess = False

  args.input = IN_PATH / args.dataset
  OUT_PATH.mkdir(parents=True, exist_ok=True)
  args.output = OUT_PATH / 'test_result'
  args.output.mkdir(parents=True, exist_ok=True)
  args.report = OUT_PATH / 'result.json'

  print('cmd_args:')
  pp(vars(args))

  run(args)
