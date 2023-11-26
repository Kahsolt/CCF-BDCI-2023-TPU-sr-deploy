#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/20 

# 测定 bmodel 推理速度

import json
from functools import reduce
from time import time

import numpy as np
from tqdm import tqdm

from run_utils import MODEL_PATH, OUT_PATH, mean
from run_bmodel import EngineOV

N_TEST = 100

fp_stats = OUT_PATH / 'stats_bmodel.json'
if fp_stats.exists():
  with open(fp_stats, 'r', encoding='utf-8') as fh:
    stats = json.load(fh)
else:
  stats = {}

fps = sorted([fp for fp in MODEL_PATH.iterdir() if fp.suffix == '.bmodel'])
for fp in tqdm(fps):
  name = fp.stem
  if name in stats:
    print(f'>> ignore {name}')
    continue
  print(f'>> benchmark {name}')

  model = EngineOV(str(fp))
  shape = model.input_shape
  numel = reduce(lambda x, y: x * y, shape)

  ts = []
  for i in tqdm(range(N_TEST)):
    x = np.random.uniform(size=shape)
    s = time()
    y = model(x)
    ts.append(time() - s)
  ts = np.asarray(ts)

  time_avg = mean(ts)
  stats[name] = {
    'input_shape': shape,
    'price': ts.mean() / numel,
    'ts_avg': ts.mean(),
    'ts_std': ts.std(),
    'ts_max': ts.max(),
    'ts_min': ts.min(),
  }
  print(stats[name])


print(f'>> save to {fp}')  
with open(fp_stats, 'w', encoding='utf-8') as fh:
  json.dump(stats, fh, indent=2, ensure_ascii=False)
