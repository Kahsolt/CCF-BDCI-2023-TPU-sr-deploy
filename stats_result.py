#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/16 

import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from run_utils import *


def run(args):
  with open(args.fp, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  if 'A' in data:
    images = data['A'][0]['images']
  else:
    images = data['B'][0]['images']

  times, niqes = [], []
  for img in images:
    times.append(img['runtime'])
    niqes.append(img['niqe'])
  times = np.asarray(times, dtype=np.float32)
  niqes = np.asarray(niqes, dtype=np.float32)

  plt.clf()
  plt.scatter(times, niqes)
  plt.xlabel('runtime')
  plt.ylabel('niqe')
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('fp', help='path to *.json file')
  args = parser.parse_args()
  
  run(args)
