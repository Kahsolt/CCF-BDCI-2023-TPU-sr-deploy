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
OUT_PATH   = BASE_PATH / 'out' ; OUT_PATH.mkdir(exist_ok=True)

# the contest scaffold
sys.path.append(str(LIB_PATH))
from fix import imgFusion2
from metrics.niqe import calculate_niqe

Box = Tuple[slice, slice]

mean = lambda x: sum(x) / len(x) if len(x) else 0.0
get_score = lambda niqe_score, i_time: math.sqrt(7 - niqe_score) / i_time * 200
