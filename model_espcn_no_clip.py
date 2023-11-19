#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/20

import torch
import torch.nn as nn
from torch import Tensor

from model_espcn import *
from model import ESPCN

MODEL_NAME = 'espcn_nc'
MODEL_SUB_PATH = MODEL_PATH / MODEL_NAME
os.makedirs(MODEL_SUB_PATH, exist_ok=True)


class ESPCN_hijack(nn.Module):
    
  def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int):
    super().__init__()

    hidden_channels = channels // 2
    out_channels = int(out_channels * (upscale_factor ** 2))

    # Feature mapping
    self.feature_maps = nn.Sequential(
      nn.Conv2d(in_channels, channels, 5, 1, 2),
      nn.Tanh(),
      nn.Conv2d(channels, hidden_channels, 3, 1, 1),
      nn.Tanh(),
    )

    # Sub-pixel convolution layer
    self.sub_pixel = nn.Sequential(
      nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
      nn.PixelShuffle(upscale_factor),
    )

  def forward(self, x: Tensor) -> Tensor:
    return self._forward_impl(x)

  # Support torch.script function.
  def _forward_impl(self, x: Tensor) -> Tensor:
    x = self.feature_maps(x)
    x = self.sub_pixel(x)
    return x


if __name__ == '__main__':
  cwd = os.getcwd()
  os.chdir(MODEL_SUB_PATH)

  ckpt = torch.load(MODEL_CKPT_FILE, map_location='cpu')
  if isinstance(ckpt, dict):
    state_dict = ckpt['state_dict']
    # ESPCN only process the Y channel in YCbCr space
    model = ESPCN_hijack(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
    model.load_state_dict(state_dict)
    example = torch.zeros([1, 1, 192, 256])
    script_model = torch.jit.trace(model, example)
    torch.jit.save(script_model, f'{MODEL_NAME}.pt')

  os.chdir(cwd)
