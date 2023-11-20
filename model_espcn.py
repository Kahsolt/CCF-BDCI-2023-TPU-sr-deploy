#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/18 

# https://github.com/Lornatang/ESPCN-PyTorch, great thanks!
# download weights, bind input_shape and convert to script_module

import sys
import os
import torch
from torch import Tensor

from run_utils import BASE_PATH, MODEL_PATH

ESPCN_PATH = BASE_PATH / 'repo' / 'ESPCN-PyTorch'
assert ESPCN_PATH.is_dir()
sys.path.append(str(ESPCN_PATH))
from model import espcn_x4
from model import ESPCN

MODEL_CKPT_FILE = MODEL_PATH / 'espcn' / 'ESPCN_x4-T91-64bf5ee4.pth.tar'
assert MODEL_CKPT_FILE.is_file(), f'please manully download the ckpt {MODEL_CKPT_FILE.name}, put at {MODEL_PATH}'


class ESPCN_nc(ESPCN):

  def _forward_impl(self, x: Tensor) -> Tensor:
    return self.sub_pixel(self.feature_maps(x))

class ESPCN_ex(ESPCN):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel '''

  def _forward_impl(self, x: Tensor) -> Tensor:
    return torch.cat([
      self.sub_pixel(self.feature_maps(x[:, 0:1, :, :])),
      self.sub_pixel(self.feature_maps(x[:, 1:2, :, :])),
      self.sub_pixel(self.feature_maps(x[:, 2:3, :, :])),
    ], dim=1)


def make_script_module(name:str):
  MODEL_SUB_PATH = MODEL_PATH / name
  os.makedirs(MODEL_SUB_PATH, exist_ok=True)

  cwd = os.getcwd()
  os.chdir(MODEL_SUB_PATH)

  ckpt = torch.load(MODEL_CKPT_FILE, map_location='cpu')
  if isinstance(ckpt, dict):
    state_dict = ckpt['state_dict']
    # ESPCN only process the Y channel in YCbCr space
    if name == 'espcn':
      model = espcn_x4(in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([1, 1, 192, 256])
    elif name == 'espcn_nc':
      model = ESPCN_nc(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([1, 1, 192, 256])
    elif name == 'espcn_ex':
      model = ESPCN_ex(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([1, 3, 192, 256])
    model.load_state_dict(state_dict)
    script_model = torch.jit.trace(model, example)
    torch.jit.save(script_model, f'{name}.pt')

  os.chdir(cwd)


if __name__ == '__main__':
  make_script_module('espcn')
  make_script_module('espcn_nc')
  make_script_module('espcn_ex')
