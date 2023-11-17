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
from PIL import Image
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
get_score = lambda niqe_score, i_time: math.sqrt(7 - niqe_score) / i_time * 200


def fix_model_size(model_size_str:str) -> Tuple[int, int]:
  if not model_size_str: return (192, 256)    # the optimal tile_size
  if ',' in model_size_str:
    return [int(e) for e in model_size_str.split(',')]
  else:
    e = int(model_size_str)
    return [e, e]


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/metrics/niqe.py
niqe_pris_params = np.load(NIQE_FILE)
mu_pris_param    = niqe_pris_params['mu_pris_param']
cov_pris_param   = niqe_pris_params['cov_pris_param']
gaussian_window  = niqe_pris_params['gaussian_window']

def get_niqe(im:ndarray) -> float:
  #assert im.dtype in [np.float32, np.float16]
  #assert im.shape[-1] == 3
  #assert 0 <= im.min() and im.max() <= 1.0
  im = bgr2ycbcr(im, y_only=True)   # [H, W], RGB => Y
  im_y = np.round(im * 255)         # float32 =? uint8
  return niqe(im_y, mu_pris_param, cov_pris_param, gaussian_window)
