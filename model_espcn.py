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

B = BATCH_SIZE
H, W = MODEL_SIZE

StateDict = Dict[str, Tensor]


class ESPCN_nc(ESPCN):

  ''' no clip '''

  def _forward_impl(self, x: Tensor) -> Tensor:
    return self.sub_pixel(self.feature_maps(x))

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

class ESPCN_ex(ESPCN):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel; no clip '''

  def _forward_impl(self, x: Tensor) -> Tensor:
    return torch.cat([
      self.sub_pixel(self.feature_maps(x[:, 0:1, :, :])),
      self.sub_pixel(self.feature_maps(x[:, 1:2, :, :])),
      self.sub_pixel(self.feature_maps(x[:, 2:3, :, :])),
    ], dim=1)

class ESPCN_ex3(ESPCN):

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

class ESPCN_ee(ESPCN_ex):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel; no clip, embed edge-enhance filter '''

  def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int):
    super().__init__(in_channels, out_channels, channels, upscale_factor)

    self.edge_enhance = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3, padding=3//2, bias=False, padding_mode='replicate')
    kernel = Tensor([
      [-1, -1, -1],
      [-1, 10, -1],
      [-1, -1, -1],
    ]).unsqueeze_(0).unsqueeze_(0).expand(3, -1, -1, -1) / 2
    self.edge_enhance.weight.data = nn.Parameter(kernel, requires_grad=False)
    self.edge_enhance.requires_grad_(False)

  def _forward_impl(self, x: Tensor) -> Tensor:
    x = super()._forward_impl(x)
    return self.edge_enhance(x)

class ESPCN_um(ESPCN_ex):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel; no clip, embed unsharp-mask filter '''

  def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int):
    super().__init__(in_channels, out_channels, channels, upscale_factor)

    # ref: https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking
    self.sharpen = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3, padding=3//2, bias=False, padding_mode='replicate')
    kernel = Tensor([
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0],
    ]).unsqueeze_(0).unsqueeze_(0).expand(3, -1, -1, -1)
    self.sharpen.weight.data = nn.Parameter(kernel, requires_grad=False)
    self.sharpen.requires_grad_(False)

  def _forward_impl(self, x: Tensor) -> Tensor:
    x = super()._forward_impl(x)
    return self.sharpen(x)

class ESPCN_approx(ESPCN):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel; replace tanh with approx, no clip '''

  def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int):
    super().__init__(in_channels, out_channels, channels, upscale_factor)

    hidden_channels = channels // 2
    out_channels = int(out_channels * (upscale_factor ** 2))

    # Feature mapping
    self.feature_maps = nn.Sequential(
      nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2), padding_mode='replicate'),
      nn.Identity(),
      nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1), padding_mode='replicate'),
      nn.Identity(),
    )

    # Sub-pixel convolution layer
    self.sub_pixel = nn.Sequential(
      nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1), padding_mode='replicate'),
      nn.PixelShuffle(upscale_factor),
    )

  def _forward_impl(self, x: Tensor) -> Tensor:
    # https://www.desmos.com/calculator
    # https://stackoverflow.com/questions/29239343/faster-very-accurate-approximation-for-tanh
    #cheap_tanh = lambda x: 2 / (1 + torch.exp(-2 * x)) - 1
    # https://mathr.co.uk/blog/2017-09-06_approximating_hyperbolic_tangent.html
    #cheap_tanh = lambda x: 3 * x / (3 + x ** 2)   # **1.44
    # just linear :)
    cheap_tanh = lambda x: x.clamp_(-1, 1)
    f = lambda x: cheap_tanh(self.feature_maps[2](cheap_tanh(self.feature_maps[0](x))))
    return torch.cat([
      self.sub_pixel(f(x[:, 0:1, :, :])),
      self.sub_pixel(f(x[:, 1:2, :, :])),
      self.sub_pixel(f(x[:, 2:3, :, :])),
    ], dim=1)

class ESPCN_approx_um(ESPCN_approx):

  ''' directly apply ESPCN to each RGB channel, even if it is pretrained in Y channel; replace tanh with approx, no clip, embed unsharp-mask filter '''

  def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int):
    super().__init__(in_channels, out_channels, channels, upscale_factor)

    # ref: https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking
    self.sharpen = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3, padding=3//2, bias=False, padding_mode='replicate')
    kernel = Tensor([
      [ 0, -1,  0],
      [-1,  5, -1],
      [ 0, -1,  0],
    ]).unsqueeze_(0).unsqueeze_(0).expand(3, -1, -1, -1)
    self.sharpen.weight.data = nn.Parameter(kernel, requires_grad=False)
    self.sharpen.requires_grad_(False)

  def _forward_impl(self, x: Tensor) -> Tensor:
    x = super()._forward_impl(x)
    return self.sharpen(x)


def state_dict_expand(state_dict:StateDict) -> StateDict:
  new_state_dict = {}
  for k, v in state_dict.items():
    ndim = len(v.shape)
    if ndim == 1:
      new_state_dict[k] = torch.tile(v, dims=[3])
    elif ndim == 4:
      new_state_dict[k] = torch.tile(v, dims=[3, 1, 1, 1])
  return new_state_dict

def state_dict_trim(state_dict:StateDict) -> StateDict:
  new_state_dict = {}
  for k, v in state_dict.items():
    if not k.endswith('.weight'):
      new_v = v
    else:
      vrng = v.max() - v.min()
      print('vrng:', vrng.item())
      if k.startswith('feature_maps'):
        thresh = 1e-2 * vrng
      else:   # sub_pixel
        thresh = 1e-3 * vrng
      print('thresh:', thresh.item())
      mask = v.abs() <= thresh
      ratio = mask.sum() / v.numel()
      print(f'>> trim {k}: {ratio: .3%}')
      new_v = torch.where(mask, torch.zeros_like(v), v)
    new_state_dict[k] = new_v
  return new_state_dict


def make_script_module(name:str):
  MODEL_SUB_PATH = MODEL_PATH / name
  os.makedirs(MODEL_SUB_PATH, exist_ok=True)

  cwd = os.getcwd()
  os.chdir(MODEL_SUB_PATH)

  ckpt = torch.load(MODEL_CKPT_FILE, map_location='cpu')
  if isinstance(ckpt, dict):
    state_dict: Dict[str, Tensor] = ckpt['state_dict']
    # original ESPCN only process the Y channel in YCbCr space
    C = 1 if name in ['espcn', 'espcn_nc'] else 3

    if name == 'espcn':
      model = espcn_x4(in_channels=1, out_channels=1, channels=64)
    elif name == 'espcn_nc':
      model = ESPCN_nc(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
    elif name == 'espcn_cp':
      model = ESPCN_cp(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
    elif name == 'espcn_ex':
      model = ESPCN_ex(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
    elif name == 'espcn_ex3':
      model = ESPCN_ex3(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      state_dict = state_dict_expand(state_dict)
    elif name == 'espcn_ee':
      model = ESPCN_ee(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
    elif name == 'espcn_um':
      model = ESPCN_um(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
    elif name == 'espcn_approx':
      model = ESPCN_approx(upscale_factor=4, in_channels=1, out_channels=1, channels=64)
      state_dict = state_dict_trim(state_dict)
    elif name == 'espcn_approx_um':
      model = ESPCN_approx_um(upscale_factor=4, in_channels=1, out_channels=1, channels=64)

    model.load_state_dict(state_dict, strict=False)
    script_model = torch.jit.trace(model, torch.zeros([B, C, H, W]))
    fn = f'{name}_{B}x{C}x{H}x{W}.pt'
    print(f'>> save to {MODEL_SUB_PATH / fn}')
    torch.jit.save(script_model, fn)

  os.chdir(cwd)


if __name__ == '__main__':
  make_script_module('espcn')
  make_script_module('espcn_nc')
  make_script_module('espcn_cp')
  make_script_module('espcn_ex')
  make_script_module('espcn_ex3')
  make_script_module('espcn_ee')
  make_script_module('espcn_um')
  make_script_module('espcn_approx')
  make_script_module('espcn_approx_um')

  os.chdir(BASE_PATH)
  os.system(f'bash ./convert.sh espcn     {B} 1 {H} {W}')
  os.system(f'bash ./convert.sh espcn_nc  {B} 1 {H} {W}')
  os.system(f'bash ./convert.sh espcn_cp  {B} 3 {H} {W}')
  os.system(f'bash ./convert.sh espcn_ex  {B} 3 {H} {W}')
  os.system(f'bash ./convert.sh espcn_ex3 {B} 3 {H} {W}')
  os.system(f'bash ./convert.sh espcn_ee  {B} 3 {H} {W}')
  os.system(f'bash ./convert.sh espcn_um  {B} 3 {H} {W}')
  os.system(f'bash ./convert.sh espcn_approx {B} 3 {H} {W}')
  os.system(f'bash ./convert.sh espcn_approx_um {B} 3 {H} {W}')
