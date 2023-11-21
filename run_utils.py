#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/17 

import warnings ; warnings.simplefilter('ignore', category=RuntimeWarning)

import os
import glob
import sys
import math
import json
from time import time
from pathlib import Path
from argparse import ArgumentParser
from typing import *

from tqdm import tqdm
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
import numpy as np
from numpy import ndarray

BASE_PATH  = Path(__file__).parent
MODEL_PATH = BASE_PATH / 'models'
if sys.platform == 'win32':     # local (develop)
  LIB_PATH = BASE_PATH / 'repo' / 'TPU-Coder-Cup' / 'CCF2023'
  IN_PATH  = BASE_PATH / 'data' / 'test'
else:                           # cloud server (deploy)
  LIB_PATH = BASE_PATH / 'TPU-Coder-Cup' / 'CCF2023'
  IN_PATH  = BASE_PATH / 'test'
NIQE_FILE  = LIB_PATH / 'metrics' / 'niqe_pris_params.npz'
OUT_PATH   = BASE_PATH / 'out' ; OUT_PATH.mkdir(exist_ok=True)

# the contest scaffold
sys.path.append(str(LIB_PATH))
from fix import imgFusion, imgFusion2
from metrics.niqe import niqe, calculate_niqe, to_y_channel
from metrics.utils import bgr2ycbcr, to_y_channel

Box = Tuple[slice, slice]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0
get_score = lambda niqe_avg, runtime_avg: math.sqrt(7 - niqe_avg) / runtime_avg * 200

POSTPROCESSOR = [
  'SHARPEN',
  'DETAIL',
  'EDGE_ENHANCE',
  'EDGE_ENHANCE_MORE',
]


def pil_to_np(img:PILImage) -> ndarray:
  return np.asarray(img, dtype=np.float32) / 255.0

def np_to_pil(im:ndarray) -> PILImage:
  return Image.fromarray(np.asarray(im * 255).astype(np.uint8))


def fix_model_size(model_size_str:str) -> Tuple[int, int]:
  if not model_size_str: return (192, 256)    # the optimal tile_size
  if ',' in model_size_str:
    return [int(e) for e in model_size_str.split(',')]
  else:
    e = int(model_size_str)
    return [e, e]


def get_parser():
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend', default='bmodel', choices=['bmodel', 'pytorch'])
  parser.add_argument('-D', '--device',  type=int,  default=0,          help='GPU/TPU device id')
  parser.add_argument('-M', '--model',   type=Path, default='r-esrgan', help='path to *.bmodel model ckpt, , or folder name under path models/')
  parser.add_argument('--model_size',    type=str,                      help='model input size like 200 or 196,256')
  parser.add_argument('--padding',       type=int,  default=16)
  parser.add_argument('--batch_size',    type=int,  default=8)
  parser.add_argument('-I', '--input',   type=Path, default=IN_PATH,    help='input image or folder')
  parser.add_argument('-L', '--limit',   type=int,  default=-1,         help='limit run sample count')
  parser.add_argument('--postprocess',   choices=POSTPROCESSOR)
  parser.add_argument('--save',          action='store_true',           help='save sr images')
  return parser

def get_args(parser:ArgumentParser=None):
  parser = parser or get_parser()
  return parser.parse_args()

def process_args(args):
  if args.backend == 'pytorch':
    suffix = '.pt'
  elif args.backend == 'bmodel':
    suffix = '.bmodel'

  fp = Path(args.model)
  if not fp.is_file():
    dp: Path = MODEL_PATH / args.model
    assert dp.is_dir(), 'should be a folder name under path models/'
    fps = [fp for fp in dp.iterdir() if fp.suffix == suffix]
    assert len(fps) == 1, f'folder contains mutiple *{suffix} files'
    args.model = fps[0]

  args.model_size = fix_model_size(args.model_size)

  args.log_dp = OUT_PATH / Path(args.model).stem
  args.log_dp.mkdir(exist_ok=True)
  args.output = args.log_dp / 'test_sr'
  args.report = args.log_dp / 'test.json'

  return args


class TiledSR:

  def __init__(self, model_size:Tuple[int, int], padding:int=4, bs:int=1):
    print('>> tiler:', self.__class__.__name__)

    self.upscale_rate = 4.0
    self.model_size = model_size  # (h, w)
    self.padding = padding
    self.bs = bs

  @property
  def tile_h(self): return self.model_size[0]
  @property
  def tile_w(self): return self.model_size[1]

  def __call__(self, im:ndarray) -> ndarray:
    raise NotImplementedError


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
      img_high = img_high.filter(getattr(ImageFilter, args.postprocess))
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

def run_eval(args, get_model:Callable, process_images:Callable):
  # in/out paths
  if Path(args.input).is_file():
    paths = [Path(args.input)]
  else:
    paths = [Path(fp) for fp in sorted(glob.glob(os.path.join(str(args.input), '*')))]
  if args.limit > 0: paths = paths[:args.limit]
  if args.save: Path(args.output).mkdir(parents=True, exist_ok=True)

  # setup model
  model = get_model(args)

  # workers & task
  start_all = time()
  niqe:    List[float] = []
  runtime: List[float] = []
  result:  List[dict]  = []
  process_images(args, model, paths, niqe, runtime, result)
  end_all = time()
  time_all = end_all - start_all
  runtime_avg = mean(runtime)
  niqe_avg = mean(niqe)
  print('time_all:',    time_all)
  print('runtime_avg:', runtime_avg)
  print('niqe_avg:',    niqe_avg)
  print('>> score:',    get_score(niqe_avg, runtime_avg))

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


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/metrics/niqe.py
niqe_pris_params = np.load(NIQE_FILE)
mu_pris_param    = niqe_pris_params['mu_pris_param']    # [1, 36]
cov_pris_param   = niqe_pris_params['cov_pris_param']   # [36, 36]
gaussian_window  = niqe_pris_params['gaussian_window']  # [7, 7]

def get_niqe(im:ndarray) -> float:
  #assert im.dtype in [np.float32, np.float16]
  #assert im.shape[-1] == 3
  #assert 0.0 <= im.min() and im.max() <= 1.0

  im_y = rgb_to_y_cb_cr(im)[0]   # [H, W], RGB => Y
  return get_niqe_y(im_y)

def get_niqe_y(im_y:ndarray) -> float:
  im_y = np.round(im_y * 255)         # float32 => uint8 in float
  return niqe(im_y, mu_pris_param, cov_pris_param, gaussian_window)

def rgb_to_y_cb_cr(im:ndarray) -> Tuple[ndarray, ndarray, ndarray]:
  ycbcr = rgb_to_ycbcr(im)
  return [ycbcr[:, :, i] for i in range(ycbcr.shape[-1])]

# repo\ESPCN-PyTorch\imgproc.py
def rgb_to_ycbcr(im:ndarray) -> ndarray:
  #assert im.dtype in [np.float32, np.float16]
  #assert im.shape[-1] == 3
  #assert 0.0 <= im.min() and im.max() <= 1.0

  w = np.asarray([
    [0.25678825, -0.14822353,  0.4392157 ],
    [0.5041294 , -0.29099217, -0.36778826],
    [0.09790588,  0.4392157 , -0.07142746],
  ], dtype=np.float32)
  b = np.asarray([
    [0.0627451, 0.5019608, 0.5019608],
  ], dtype=np.float32)
  return np.matmul(im, w) + b   # float32, should be in 0~1

# repo\ESPCN-PyTorch\imgproc.py
def ycbcr_to_rgb(im:ndarray) -> ndarray:
  #assert im.dtype in [np.float32, np.float16]
  #assert im.shape[-1] == 3

  w = np.asarray([
    [1.16438355,  1.16438355, 1.16438355],
    [0.        , -0.3917616 , 2.01723105],
    [1.59602715, -0.81296805, 0.        ],
  ], dtype=np.float32)
  b = np.asarray([
    [-0.87420005,  0.53167063, -1.0856314],
  ], dtype=np.float32)
  return np.matmul(im, w) + b   # float32, should be in 0~1


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  def stats(x:ndarray, name:str): print(name, x.min(), x.max(), x.mean(), x.std())

  img = Image.open(IN_PATH / '0001.png')
  rgb = pil_to_np(img)
  stats(rgb, 'rgb')
  ycbcr = rgb_to_ycbcr(rgb)
  stats(ycbcr, 'ycbcr')
  rgb_hat = ycbcr_to_rgb(ycbcr)
  stats(rgb_hat, 'rgb_hat')
  print('L1:', np.mean(np.abs(rgb_hat - rgb)))
  print('L2:', np.mean(np.abs(rgb_hat - rgb)**2))
