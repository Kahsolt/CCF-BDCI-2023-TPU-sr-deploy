#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/16 

# https://github.com/Coloquinte/torchSR, great thanks!
# download weights, bind input_shape and convert to script_module

from pathlib import Path

BASE_PATH = Path(__file__).parent
BUILD_PATH = BASE_PATH / 'build'
OUT_FILE = BUILD_PATH / 'ninasr.pt'
if OUT_FILE.is_file(): exit(0)


import sys
import os
import torch

cwd = os.getcwd()
BUILD_PATH.mkdir(exist_ok=True)
os.chdir(BUILD_PATH)

TORCHSR_PATH = BUILD_PATH / 'torchSR'
if not TORCHSR_PATH.is_dir():
  os.system('git clone https://github.com/Coloquinte/torchSR')
assert TORCHSR_PATH.is_dir()
sys.path.append(str(TORCHSR_PATH))
from torchsr.models import ninasr_b0

MODEL_FILE = BUILD_PATH / 'ninasr_b0_x4.pt'
if not MODEL_FILE.is_file():
  os.system('wget -nc https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x4.pt')
assert MODEL_FILE.is_file()

state_dict = torch.load(MODEL_FILE, map_location='cpu')
model = ninasr_b0(scale=4, pretrained=False)
model.load_state_dict(state_dict)
example = torch.zeros([1, 3, 192, 256])
script_model = torch.jit.trace(model, example)
torch.jit.save(script_model, str(OUT_FILE.name))

os.chdir(cwd)
