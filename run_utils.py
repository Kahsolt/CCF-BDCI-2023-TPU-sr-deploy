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
from metrics.utils import bgr2ycbcr

Box = Tuple[slice, slice]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0
get_score = lambda niqe_avg, runtime_avg: math.sqrt(7 - niqe_avg) / runtime_avg * 200


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


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/metrics/niqe.py
niqe_pris_params = np.load(NIQE_FILE)
mu_pris_param    = niqe_pris_params['mu_pris_param']    # [1, 36]
cov_pris_param   = niqe_pris_params['cov_pris_param']   # [36, 36]
gaussian_window  = niqe_pris_params['gaussian_window']  # [7, 7]

def get_niqe(im:ndarray) -> float:
  #assert im.dtype in [np.float32, np.float16]
  #assert im.shape[-1] == 3
  #assert 0 <= im.min() and im.max() <= 1.0
  im = rgb2bgr(im)                    # rgb2bgr, float32
  im_y = bgr2ycbcr(im, y_only=True)   # [H, W], RGB => Y
  im_y = np.round(im_y * 255)         # float32 => uint8 in float
  return niqe(im_y, mu_pris_param, cov_pris_param, gaussian_window)

def rgb2bgr(im:ndarray) -> ndarray:
  return im[:, :, ::-1]

bgr2rgb = rgb2bgr

def get_y_cb_cr(im:ndarray) -> Tuple[ndarray, ndarray, ndarray]:
  im = rgb2bgr(im)
  ycbcr: ndarray = bgr2ycbcr(im, y_only=False)
  return [ycbcr[:, :, i] for i in range(ycbcr.shape[-1])]

# repo\ESPCN-PyTorch\imgproc.py
def ycbcr_to_bgr(img:ndarray) -> ndarray:
  dtype = img.dtype
  img *= 255.
  w = np.asarray([
    [0.00456621, 0.00456621, 0.00456621],
    [0.00791071, -0.00153632, 0],
    [0, -0.00318811, 0.00625893],
  ])
  b = [-276.836, 135.576, -222.921]
  img = np.matmul(img, w) * 255.0 + b
  img /= 255.
  img = img.astype(dtype)
  return img
