#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/16 

from code import interact
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.nn import Module


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('fp', type=Path, help='path to *.pt model ckpt')
  args = parser.parse_args()

  fp = Path(args.fp)
  assert fp.is_file(), 'fp must be a file'


  model: Module = torch.load(fp, map_location='cpu')
  model = model.eval()
  model_str = str(model)
  print(model_str)
  param_cnt = sum([p.numel() for p in model.parameters() if p.requires_grad])
  dtype = list(model.parameters())[0].dtype
  print(f'param_cnt: {param_cnt} ({param_cnt/10**6:.3f} M) in dtype {dtype}')

  fp_out = fp.with_suffix('.arch')
  if not fp_out.exists():
    print(f'>> write to {fp_out}')
    with open(fp_out, 'w', encoding='utf-8') as fh:
      fh.write(f'param_cnt: {param_cnt}\n')
      fh.write('\n')
      fh.write(model_str)

  X = torch.rand([1, 3, 224, 224], dtype=dtype)

  interact(local=globals())
