#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/16 

# https://github.com/Coloquinte/torchSR, great thanks!
# download weights, bind input_shape and convert to script_module

import sys
import os
from pathlib import Path
from collections import OrderedDict
import torch

from run_utils import BASE_PATH, MODEL_PATH

TORCHSR_PATH = BASE_PATH / 'repo' / 'torchSR'
assert TORCHSR_PATH.is_dir()
sys.path.append(str(TORCHSR_PATH))
from torchsr.models import carn, carn_m, edsr_r16f64, ninasr_b0, ninasr_b1, ninasr_b2, rcan

# only x4 models
MODEL_URLS = [
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/carn.pt',          # 6.09 MB
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/carn_m.pt',        # 1.59 MB
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr64_x4.pt',     # 5.8 MB
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/edsr_x4.pt',       # 164 MB
  'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x4.pt',  # 428 KB
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b1_x4.pt',  # 3.97 MB
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b2_x4.pt',  # 38.5 MB
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/rcan_x4.pt',       # 59.8 MB
# 'https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/rdn_x4.pt',        # 85 MB
]

CLASS_MAP = {
  'carn':         carn,
  'carn_m':       carn_m,
  'edsr64_x4':    edsr_r16f64,
  'ninasr_b0_x4': ninasr_b0,
  'ninasr_b1_x4': ninasr_b1,
  'ninasr_b2_x4': ninasr_b2,
  'rcan_x4':      rcan,
}


def run(cmd:str):
  print(f'[run] {cmd}')
  os.system(cmd)


if __name__ == '__main__':
  os.chdir(MODEL_PATH)
  model_names = [dp.stem for dp in MODEL_PATH.iterdir() if dp.is_dir()]
  for url in MODEL_URLS:
    cwd = os.getcwd()

    name = Path(url).stem
    if name not in model_names: os.mkdir(name)
    os.chdir(name)

    model_file = Path(url).name
    if not Path(model_file).exists():
      run(f'wget {url}')

    state_dict = torch.load(model_file, map_location='cpu')
    if isinstance(state_dict, OrderedDict):
      model = CLASS_MAP[name](scale=4, pretrained=False)
      model.load_state_dict(state_dict)
      example = torch.zeros([1, 3, 192, 256])
      script_model = torch.jit.trace(model, example)
      torch.jit.save(script_model, model_file)

    os.chdir(cwd)
