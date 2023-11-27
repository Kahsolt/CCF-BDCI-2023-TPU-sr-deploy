#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/27 

# 测定 bmodel 推理速度 (多模型多线程)

import math
from threading import Thread
from typing import List, Tuple

from stats_bmodel import *

N_TEST = 1000

fp_stats = OUT_PATH / 'stats_bmodel_mt.json'
if fp_stats.exists():
  with open(fp_stats, 'r', encoding='utf-8') as fh:
    stats = json.load(fh)
else:
  stats = {}

fps = sorted([fp for fp in MODEL_PATH.iterdir() if fp.suffix == '.bmodel'])
fp = fps[0]
name = fp.stem
stats['name'] = name
stats['shape'] = None
stats['mthread'] = {}
print(f'>> benchmark {name}')


def task(idxs:range, shape:Tuple[int], model, ts:List[float]):
  for i in idxs:
    if i>= N_TEST: break
    x = np.random.uniform(size=shape)
    s = time()
    y = model(x)
    ts[i] = time() - s

for n_worker in tqdm([1, 2, 4, 8]):
  models = [EngineOV(str(fp)) for _ in range(n_worker)]
  shape = models[0].input_shape
  stats['shape'] = shape
  numel = reduce(lambda x, y: x * y, shape)
  n_jobs = math.ceil(N_TEST / n_worker)

  ts = [None] * N_TEST
  thrs = [Thread(target=task, args=(range(i*n_jobs, (i+1)*n_jobs), shape, model, ts)) for i, model in enumerate(models)]
  for thr in thrs: thr.start()
  for thr in thrs: thr.join()
  ts = np.asarray(ts)

  time_avg = mean(ts)

  stats[n_worker] = {
    'price': ts.mean() / numel,
    'ts_avg': ts.mean(),
    'ts_std': ts.std(),
    'ts_max': ts.max(),
    'ts_min': ts.min(),
  }
  print(stats)

print(f'>> save to {fp_stats}')  
with open(fp_stats, 'w', encoding='utf-8') as fh:
  json.dump(stats, fh, indent=2, ensure_ascii=False)
