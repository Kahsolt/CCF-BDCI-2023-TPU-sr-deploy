#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/18 

# https://github.com/Lornatang/ESPCN-PyTorch, great thanks!
# download weights, bind input_shape and convert to script_module

import sys
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from run_utils import BASE_PATH, MODEL_PATH, MODEL_SIZE, BATCH_SIZE

ESPCN_PATH = BASE_PATH / 'repo' / 'ESPCN-PyTorch'
assert ESPCN_PATH.is_dir()
sys.path.append(str(ESPCN_PATH))
from model import espcn_x4
from model import ESPCN

MODEL_CKPT_FILE = MODEL_PATH / 'espcn' / 'ESPCN_x4-T91-64bf5ee4.pth.tar'
assert MODEL_CKPT_FILE.is_file(), f'please manully download the ckpt {MODEL_CKPT_FILE.name}, put at {MODEL_PATH}'


class ESPCN_nc(ESPCN):

  ''' no clip '''

  def _forward_impl(self, x: Tensor) -> Tensor:
    return self.sub_pixel(self.feature_maps(x))

class ESPCN_ex(ESPCN):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel; no clip '''

  def _forward_impl(self, x: Tensor) -> Tensor:
    return torch.cat([
      self.sub_pixel(self.feature_maps(x[:, 0:1, :, :])),
      self.sub_pixel(self.feature_maps(x[:, 1:2, :, :])),
      self.sub_pixel(self.feature_maps(x[:, 2:3, :, :])),
    ], dim=1)

class ESPCN_ex3(ESPCN_nc):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel via group-conv2d; no clip '''

  def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int):
    super(ESPCN, self).__init__()

    hidden_channels = channels // 2
    out_channels = int(out_channels * (upscale_factor ** 2))

    # Feature mapping
    self.feature_maps = nn.Sequential(
      nn.Conv2d(in_channels*3, channels*3, (5, 5), (1, 1), (2, 2), groups=3),
      nn.Tanh(),
      nn.Conv2d(channels*3, hidden_channels*3, (3, 3), (1, 1), (1, 1), groups=3),
      nn.Tanh(),
    )

    # Sub-pixel convolution layer
    self.sub_pixel = nn.Sequential(
      nn.Conv2d(hidden_channels*3, out_channels*3, (3, 3), (1, 1), (1, 1), groups=3),
      nn.PixelShuffle(upscale_factor),
    )

  def _forward_impl(self, x: Tensor) -> Tensor:
    fmaps = self.feature_maps(x)    # [B=4, C=96, H=192, W=256]
    x = self.sub_pixel[0](fmaps)    # [B=4, C=48, H=192, W=256]
    pshuff = self.sub_pixel[1]
    return torch.cat([
      pshuff(x[:,  0:16, :, :]),
      pshuff(x[:, 16:32, :, :]),
      pshuff(x[:, 32:48, :, :]),
    ], dim=1)

class ESPCN_cp(ESPCN):

  ''' transform to YCbCr, apply ESPCN to Y channel, transform back to RGB; no clip '''

  def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int):
    super().__init__(in_channels, out_channels, channels, upscale_factor)

    self.rgb2ycbcr = nn.Linear(3, 3)
    self.rgb2ycbcr.weight.data = nn.Parameter(Tensor([
      [0.25678825, -0.14822353,  0.4392157 ],
      [0.5041294 , -0.29099217, -0.36778826],
      [0.09790588,  0.4392157 , -0.07142746],
    ]).T, requires_grad=False)
    self.rgb2ycbcr.bias.data = nn.Parameter(Tensor([
      [0.0627451, 0.5019608, 0.5019608],
    ]), requires_grad=False)
    self.rgb2ycbcr.requires_grad_(False)

    self.ycbcr2rgb = nn.Linear(3, 3)
    self.ycbcr2rgb.weight.data = nn.Parameter(Tensor([
      [1.16438355,  1.16438355, 1.16438355],
      [0.        , -0.3917616 , 2.01723105],
      [1.59602715, -0.81296805, 0.        ],
    ]).T, requires_grad=False)
    self.ycbcr2rgb.bias.data = nn.Parameter(Tensor([
      [-0.87420005,  0.53167063, -1.0856314],
    ]), requires_grad=False)
    self.ycbcr2rgb.requires_grad_(False)

    if 'nearest':
      self.up = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4, bias=False)
      w = torch.ones_like(self.up.weight)
    self.up.weight.data = nn.Parameter(w, requires_grad=False)
    self.up.requires_grad_(False)

  def _forward_impl(self, x: Tensor) -> Tensor:
    # RGB to YCbCr
    x = x.permute([0, 2, 3, 1])   # bchw => bhwc
    z = self.rgb2ycbcr(x)
    z = z.permute([0, 3, 1, 2])   # bhwc => bchw
    # up each channel
    o = torch.cat([
      self.sub_pixel(self.feature_maps(z[:, 0:1, :, :])),
      self.sub_pixel(self.feature_maps(z[:, 1:2, :, :])),
      self.sub_pixel(self.feature_maps(z[:, 2:3, :, :])),
      #self.up(z[:, 1:2, :, :]),
      #self.up(z[:, 2:3, :, :]),
      #F.interpolate(z[:, 1:2, :, :], scale_factor=4, mode='bilinear'),
      #F.interpolate(z[:, 2:3, :, :], scale_factor=4, mode='bilinear'),
    ], dim=1)
    # YCbCr to RGB
    o = o.permute([0, 2, 3, 1])
    y = self.ycbcr2rgb(o)
    y = y.permute([0, 3, 1, 2])
    return y


def make_script_module(name:str):
  MODEL_SUB_PATH = MODEL_PATH / name
  os.makedirs(MODEL_SUB_PATH, exist_ok=True)

  cwd = os.getcwd()
  os.chdir(MODEL_SUB_PATH)

  ckpt = torch.load(MODEL_CKPT_FILE, map_location='cpu')
  if isinstance(ckpt, dict):
    state_dict: Dict[str, Tensor] = ckpt['state_dict']
    if name == 'espcn':       # ESPCN only process the Y channel in YCbCr space
      model = espcn_x4(in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([BATCH_SIZE, 1, *MODEL_SIZE])
    elif name == 'espcn_nc':  # ESPCN only process the Y channel in YCbCr space
      model = ESPCN_nc(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([BATCH_SIZE, 1, *MODEL_SIZE])
    elif name == 'espcn_ex':
      model = ESPCN_ex(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([BATCH_SIZE, 3, *MODEL_SIZE])
    elif name == 'espcn_ex3':
      model = ESPCN_ex3(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([BATCH_SIZE, 3, *MODEL_SIZE])

      if 'repeat weights':
        new_state_dict = {}
        for k, v in state_dict.items():
          ndim = len(v.shape)
          if ndim == 1:
            new_state_dict[k] = torch.tile(v, dims=[3])
          elif ndim == 4:
            new_state_dict[k] = torch.tile(v, dims=[3, 1, 1, 1])
        state_dict = new_state_dict   # replace

    elif name == 'espcn_cp':
      model = ESPCN_cp(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      example = torch.zeros([BATCH_SIZE, 3, *MODEL_SIZE])
    model.load_state_dict(state_dict, strict=False)
    script_model = torch.jit.trace(model, example)
    print(f'>> save to {MODEL_SUB_PATH / name}_{BATCH_SIZE}.pt')
    torch.jit.save(script_model, f'{name}_{BATCH_SIZE}.pt')

  os.chdir(cwd)


if __name__ == '__main__':
  make_script_module('espcn')
  make_script_module('espcn_nc')
  make_script_module('espcn_ex')
  make_script_module('espcn_ex3')
  make_script_module('espcn_cp')

  os.system('bash ./convert_espcn.sh')
