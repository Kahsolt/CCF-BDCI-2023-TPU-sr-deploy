#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/18 

# https://github.com/Lornatang/FSRCNN-PyTorch, great thanks!
# download weights, bind input_shape and convert to script_module

import sys
import os
import torch

from run_utils import BASE_PATH, MODEL_PATH

FSRCNN_PATH = BASE_PATH / 'repo' / 'FSRCNN-PyTorch'
assert FSRCNN_PATH.is_dir()
sys.path.append(str(FSRCNN_PATH))
from model import FSRCNN

MODEL_NAME = 'fsrcnn'
MODEL_CKPT_FILE = MODEL_PATH / MODEL_NAME / 'fsrcnn_x4-T91-97a30bfb.pth.tar'
assert MODEL_CKPT_FILE.is_file(), f'please manully download the ckpt {MODEL_CKPT_FILE.name}, put at {MODEL_PATH}'


if __name__ == '__main__':
  cwd = os.getcwd()
  os.chdir(MODEL_PATH / MODEL_NAME)

  ckpt = torch.load(MODEL_CKPT_FILE, map_location='cpu')
  if isinstance(ckpt, dict):
    state_dict = ckpt['state_dict']
    # FSRCNN only process the Y channel in YCbCr space
    model = FSRCNN(upscale_factor=4)
    model.load_state_dict(state_dict)
    example = torch.zeros([1, 1, 192, 256])
    script_model = torch.jit.trace(model, example)
    torch.jit.save(script_model, f'{MODEL_NAME}.pt')

  os.chdir(cwd)
