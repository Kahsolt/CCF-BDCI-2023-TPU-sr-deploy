#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/17 

import os
import torch
import torch.nn as nn
from torch import Tensor

from run_utils import MODEL_PATH

# dummy models testing the TPU computational capacity


class EmptyModel(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x:Tensor) -> Tensor:
    x = torch.cat([x, x, x, x], dim=-2)
    x = torch.cat([x, x, x, x], dim=-1)
    return x


class CheapModel(nn.Module):

  def __init__(self):
    super().__init__()

    # nearest upsample
    self.up = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=4, groups=3, bias=False)
    self.up.weight.data = nn.Parameter(torch.ones_like(self.up.weight))
    self.up.requires_grad_(False)

    # detail filter enhance
    self.filter = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
    kernel = Tensor([
      [0,  -1,  0],
      [-1, 10, -1],
      [0,  -1,  0],
    ]).unsqueeze_(0).unsqueeze_(0).expand((3, 1, -1, -1)) / 6
    self.filter.weight.data = nn.Parameter(kernel)
    self.filter.requires_grad_(False)

  def forward(self, x:Tensor) -> Tensor:
    x = self.up(x)
    x = self.filter(x)
    return x


def convert_script_model(model_cls:type, name:str):
  model = model_cls()
  x = torch.zeros([4, 3, 192, 256])
  print(x.shape)
  x_hat = model(x)
  print(x_hat.shape)

  script_model = torch.jit.trace(model, x)
  dp = MODEL_PATH / name
  dp.mkdir(exist_ok=True, parents=True)

  print(f'>> saving to folder {dp}')
  cwd = os.getcwd()
  os.chdir(dp)
  torch.jit.save(script_model, f'{name}.pt')
  os.chdir(cwd)


if __name__ == '__main__':
  convert_script_model(EmptyModel, 'empty')
  convert_script_model(CheapModel, 'cheap')
