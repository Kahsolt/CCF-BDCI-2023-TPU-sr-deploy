#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/16 

# this script should run in docker
# compile r-esrgan4x+ with inputs size as the optimal tile size estimated by `stats_dataset.py`

import os
from pathlib import Path

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data' / 'test'
MODEL_PATH = BASE_PATH / 'models'
MODEL_FILE = MODEL_PATH / 'r-esrgan/r-esrgan4x+.pt'
assert MODEL_FILE.is_file(), '>> run "convert_resrgan.sh" in docker first'
OUT_PATH = MODEL_PATH / 'r-esrgan.tile' ; OUT_PATH.mkdir(parents=True, exist_ok=True)
os.chdir(OUT_PATH)


MODEL_DEVICE = 'bm1684x'
MODEL_DTYPE = [ 'F32', 'BF16', 'F16', 'INT8', 'INT4', 'QDQ' ]

CMD_TRANSFORM_MLIR = f'''
model_transform.py 
 --model_name r-esrgan 
 --input_shape [[1,3,192,256]] 
 --model_def "{str(MODEL_FILE)}" 
 --mlir r-esrgan4x.mlir
'''.replace('\n', '').replace('  ', ' ')

CMD_RUN_CALIB = f'''
run_calibration.py 
  r-esrgan4x.mlir
  --dataset "{str(DATA_PATH)}"
  --input_num 100
  -o r-esrgan4x.DTYPE.cali
'''.replace('\n', '').replace('  ', ' ')

CMD_DEPLOY_BMODEL = f'''
model_deploy.py 
 --mlir r-esrgan4x.mlir 
 --quantize DTYPE 
 --chip {MODEL_DEVICE} 
 --model r-esrgan4x.DTYPE.bmodel
'''.replace('\n', '').replace('  ', ' ')

CMD_DEPLOY_BMODEL_WITH_CALIB = f'''
model_deploy.py 
  --mlir r-esrgan4x.mlir 
  --quantize DTYPE 
  --calibration_table r-esrgan4x.DTYPE.cali 
  --chip {MODEL_DEVICE} 
  --tolerance 0.85,0.45 \
  --model r-esrgan4x.DTYPE.bmodel
'''.replace('\n', '').replace('  ', ' ')


def run(cmd:str):
  print(f'[run] {cmd}')
  os.system(cmd)

def bind_args(cmd:str, dtype:float=None):
  if dtype: cmd = cmd.replace('DTYPE', dtype)
  return cmd


if __name__ == '__main__':
  fn = f'r-esrgan4x.mlir'
  if not Path(fn).exists():
    run(bind_args(CMD_TRANSFORM_MLIR))

  for dtype in MODEL_DTYPE:
    if dtype.startswith('INT'):
      fn = f'r-esrgan4x.{dtype}.cali'
      if not Path(fn).exists():
        run(bind_args(CMD_RUN_CALIB, dtype))
      fn = f'r-esrgan4x.{dtype}.bmodel'
      if not Path(fn).exists():
        run(bind_args(CMD_DEPLOY_BMODEL_WITH_CALIB, dtype))
    else:
      fn = f'r-esrgan4x.{dtype}.bmodel'
      if not Path(fn).exists():
        run(bind_args(CMD_DEPLOY_BMODEL, dtype))
