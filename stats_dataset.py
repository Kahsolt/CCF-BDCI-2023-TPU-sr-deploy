#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/16 

import math
import json
from argparse import ArgumentParser
from PIL import Image

from scipy.stats import mode
from scipy.optimize import differential_evolution, shgo, dual_annealing, brute
import matplotlib.pyplot as plt

from run_utils import *

pad_size = 0


def gather_hws(dataset:str) -> Tuple[List[int], List[int]]:
  dp_in = IN_PATH / dataset
  dp_out = OUT_PATH / dataset
  dp_out.mkdir(exist_ok=True)

  fp = dp_out / 'dataset_sizes.json'
  if not fp.exists():
    hs, ws = [], []
    for img_fp in dp_in.iterdir():
      img = Image.open(img_fp)
      w, h = img.size   # NOTE: mind the order
      hs.append(h)
      ws.append(w)

    print(f'>> save to {fp}')
    with open(fp, 'w', encoding='utf-8') as fh:
      data = { 'hs': hs, 'ws': ws }
      json.dump(data, fh, indent=2, ensure_ascii=False)

  print(f'>> load from {fp}')
  with open(fp, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
    hs, ws = data['hs'], data['ws']

  print(f'[height] mode: {mode(hs).mode}, mean: {mean(hs)}')
  print(f'[width]  mode: {mode(ws).mode}, mean: {mean(ws)}')

  fp = dp_out / 'dataset_hw_hist.png'
  if not fp.exists():
    plt.clf()
    plt.subplot(121) ; plt.title('h') ; plt.hist(hs, bins=32)
    plt.subplot(122) ; plt.title('w') ; plt.hist(ws, bins=32)
    plt.tight_layout()
    print(f'>> save to {fp}')
    plt.savefig(fp, dpi=600)

  fp = dp_out / 'dataset_size_hist.png'
  if not fp.exists():
    plt.clf()
    plt.hist(hs + ws, bins=32)
    plt.tight_layout()
    print(f'>> save to {fp}')
    plt.savefig(fp, dpi=600)

  return hs, ws


def find_optimal_tile_size(hs:List[int], ws:List[int]):
  def func(x:list) -> int:
    tile_h, tile_w = x
    cost = 0
    for h, w in zip(hs, ws):
      n_h = math.ceil(h / (tile_h - pad_size))
      n_w = math.ceil(w / (tile_w - pad_size))
      tile_cnt = n_h * n_w
      area = h * w
      area_tiled = tile_cnt * (tile_h * tile_w)
      blank_ratio = 1 - area / area_tiled
      # we want both small `blank_ratio` and `tile_cnt`
      cost += blank_ratio * (tile_cnt + 1)
    return cost

  bounds = [(160, 320), (160, 320)]

  res = shgo(func, bounds, n=200)
  print('[shgo]')
  print('  tile_size = ', res.x)
  print('  cost = ', res.fun)
  res = dual_annealing(func, bounds)
  print('[dual_annealing]')
  print('  tile_size = ', res.x)
  print('  cost = ', res.fun)
  res = differential_evolution(func, bounds, popsize=200)
  print('[differential_evolution]')
  print('  tile_size = ', res.x)
  print('  cost = ', res.fun)

  res = brute(func, ranges=bounds)
  print('[brute]')
  print('  tile_size = ', res)


def get_avg_tile_count(hs:List[int], ws:List[int], optimal_tile_size:Tuple[int, int]):
  optimal_tile_size = (192, 256)
  tile_h, tile_w = optimal_tile_size

  tile_cnt = []
  for h, w in zip(hs, ws):
    n_h = math.ceil(h / (tile_h - pad_size))
    n_w = math.ceil(w / (tile_w - pad_size))
    tile_cnt.append(n_h * n_w)

  print('avg_tile_count:', mean(tile_cnt))


def run(args):
  hs, ws = gather_hws(args.dataset)
  
  find_optimal_tile_size(hs, ws)
  optimal_tile_size = (192, 256)

  get_avg_tile_count(hs, ws, optimal_tile_size)
  avg_tile_count = 10.125


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--dataset', required=True, choices=DATASETS, help='dataset slpit')
  args = parser.parse_args()

  run(args)
