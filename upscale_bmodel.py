import sys
import os
import glob
import math
import json
from time import time
from threading import Thread, Event, RLock
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage
from argparse import ArgumentParser
import warnings ; warnings.simplefilter('ignore', category=RuntimeWarning)
from typing import List

import numpy as np
from numpy import ndarray
from tqdm import tqdm

BASE_PATH = Path(__file__).parent
MODEL_FILE = BASE_PATH / 'model.bmodel'
LIB_PATH = BASE_PATH / 'TPU-Coder-Cup' / 'CCF2023'
IN_PATH = BASE_PATH / 'test'
OUT_PATH = BASE_PATH / 'out' ; OUT_PATH.mkdir(exist_ok=True)
IMAGE_PATH = OUT_PATH / 'test_sr'
REPORT_FILE = OUT_PATH / 'test.json'

# TPU engine sdk
import sophon.sail as sail
sail.set_print_flag(False)
sail.set_dump_io_flag(False)
# contest scaffolds
sys.path.append(str(LIB_PATH))
from fix import imgFusion2
from metrics.niqe import calculate_niqe

mean = lambda x: sum(x) / len(x) if len(x) else 0.0


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/fix.py
def imgFusion(img_list, overlap, res_w, res_h):
  pre_v_img = None
  for vi in range(len(img_list)):
    h_img = np.transpose(img_list[vi][0], (1,2,0))
    for hi in range(1, len(img_list[vi])):
      new_img = np.transpose(img_list[vi][hi], (1,2,0))
      h_img = imgFusion2(h_img, new_img, (h_img.shape[1]+new_img.shape[1]-res_w) if (hi == len(img_list[vi])-1) else overlap, True)
    pre_v_img = h_img if pre_v_img is None else imgFusion2(pre_v_img, h_img, (pre_v_img.shape[0]+h_img.shape[0]-res_h) if vi == len(img_list)-1 else overlap, False)
  return np.transpose(pre_v_img, (2,0,1))


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/npuengine.py
class EngineOV:

  def __init__(self, model_path:str, device_id:int=0):
    if 'DEVICE_ID' in os.environ:
      device_id = int(os.environ['DEVICE_ID'])
      print('>> device_id is in os.environ. and device_id = ', device_id)
    try:
      self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
    except Exception as e:
      print('load model error; please check model path and device status')
      print('>> model_path: ', model_path)
      print('>> device_id: ', device_id)
      print('>> sail.Engine error: ', e)
      raise e

    self.model_path  = model_path
    self.device_id   = device_id
    self.graph_name  = self.model.get_graph_names()[0]
    self.input_name  = self.model.get_input_names(self.graph_name)
    self.output_name = self.model.get_output_names(self.graph_name)

  def __str__(self):
    return f'EngineOV: model_path={self.model_path}, device_id={self.device_id}'

  def __call__(self, values:list):
    assert isinstance(values, list), 'input should be a list'
    input = { self.input_name[i]: values[i] for i in range(len(values)) }
    output = self.model.process(self.graph_name, input)
    return [output[name] for name in self.output_name]


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/upscale.py
class UpscaleModel:

  def __init__(self, model_fp, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=4, device_id=0):
    self.model = EngineOV(str(model_fp), device_id)
    self.model_size = model_size
    self.upscale_rate = upscale_rate
    self.tile_size = tile_size
    self.padding = padding

  def calc_tile_position(self, width, height, col, row):
    # generate mask
    tile_left   = col * self.tile_size[0]
    tile_top    = row * self.tile_size[1]
    tile_right  = (col + 1) * self.tile_size[0] + self.padding
    tile_bottom = (row + 1) * self.tile_size[1] + self.padding
    if tile_right > height:
      tile_right = height
      tile_left = height - self.tile_size[0] - self.padding * 1
    if tile_bottom > width:
      tile_bottom = width
      tile_top = width - self.tile_size[1] - self.padding * 1
    return tile_top, tile_left, tile_bottom, tile_right

  def calc_upscale_tile_position(self, tile_left, tile_top, tile_right, tile_bottom):
    return [int(e * self.upscale_rate) for e in (tile_left, tile_top, tile_right, tile_bottom)]

  def model_process(self, tile:PILImage):
    # preprocess
    ntile = tile.resize(self.model_size)  # (216, 216) => (200, 200)
    ntile = np.asarray(ntile).astype(np.float32)
    ntile = ntile / 255
    ntile = np.transpose(ntile, (2, 0, 1))
    ntile = ntile[np.newaxis, :, :, :]    # [B=1, C=3, H=200, W=200]
    # model forward
    res: ndarray = self.model([ntile])[0][0]
    res = res.clip(0.0, 1.0)
    # extract padding
    res = np.transpose(res, (1, 2, 0))
    res = res * 255
    res = res.astype(np.uint8)
    res = Image.fromarray(res)
    res = res.resize(self.target_tile_size)   # (800, 800) => (864, 864)
    return res

  def extract_and_enhance_tiles(self, image:PILImage, upscale_ratio:float=2.0) -> PILImage:
    if image.mode != 'RGB':
      image = image.convert('RGB')
    # 获取图像的宽度和高度
    width, height = image.size
    self.upscale_rate = upscale_ratio
    self.target_tile_size = (int((self.tile_size[0] + self.padding * 1) * upscale_ratio), int((self.tile_size[1] + self.padding * 1) * upscale_ratio))
    target_width, target_height = int(width * upscale_ratio), int(height * upscale_ratio)
    # 计算瓦片的列数和行数
    num_cols = math.ceil((width  - self.padding) / self.tile_size[0])
    num_rows = math.ceil((height - self.padding) / self.tile_size[1])

    # 遍历每个瓦片的行和列索引
    img_tiles = []
    for row in range(num_rows):
      img_h_tiles = []
      for col in range(num_cols):
        # 计算瓦片的左上角和右下角坐标
        tile_left, tile_top, tile_right, tile_bottom = self.calc_tile_position(width, height, row, col)
        # 裁剪瓦片
        tile = image.crop((tile_left, tile_top, tile_right, tile_bottom))
        # 使用超分辨率模型放大瓦片
        upscaled_tile = self.model_process(tile)
        # 将放大后的瓦片粘贴到输出图像上
        # overlap
        ntile = np.asarray(upscaled_tile).astype(np.float32)
        ntile = np.transpose(ntile, (2, 0, 1))
        img_h_tiles.append(ntile)
      img_tiles.append(img_h_tiles)
    res = imgFusion(img_list=img_tiles, overlap=int(self.padding * upscale_ratio), res_w=target_width, res_h=target_height)
    res = Image.fromarray(np.transpose(res, (1, 2, 0)).astype(np.uint8))
    return res


def worker(thr_id:int, args, paths:List[Path], result:List[dict], runtime:List[float], niqe:List[float], is_stop:Event=None, lock:RLock=None):
  upmodel = UpscaleModel(args.model, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=20, device_id=thr_id)

  total = len(paths)
  for idx, fp in enumerate((tqdm if thr_id == 0 else list)(paths)):
    if is_stop.is_set(): return

    # 加载图片
    img = Image.open(fp)

    # 模型推理
    start = time()
    res = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0)
    end = time() - start
    
    if lock: lock.acquire()
    runtime.append(end)
    if lock: lock.release()

    # 保存图片
    if args.save:
      fp_out = Path(args.output) / fp.name
      res.save(fp_out)
      img = Image.open(fp_out)
    else:
      img = res

    # 计算niqe
    output = np.asarray(img)
    niqe_output = calculate_niqe(output, 0, input_order='HWC', convert_to='y')

    if lock: lock.acquire()
    niqe.append(niqe_output)
    result.append({'img_name': fp.stem, 'runtime': format(end, '.4f'), 'niqe': format(niqe_output, '.4f')})
    if lock: lock.release()

    if thr_id == 0 and (idx + 1) % 10 == 0:
      print(f'>> [{idx+1}/{total}]: niqe {mean(niqe)}, time {mean(runtime)}')


def run(args):
  # in/out paths
  if Path(args.input).is_file():
    paths = [Path(args.input)]
  else:
    paths = [Path(fp) for fp in sorted(glob.glob(os.path.join(str(args.input), '*')))]
  if args.limit > 0: paths = paths[:args.limit]
  if args.save: Path(args.output).mkdir(parents=True, exist_ok=True)

  # workers & task
  is_stop = Event()
  result:  List[dict]  = []
  runtime: List[float] = []
  niqe:    List[float] = []

  start_all = time()
  if args.n_worker == 0:
    try:
      worker(0, args, paths, result, runtime, niqe, is_stop, None)
    except KeyboardInterrupt:
      print('Exit by Ctrl+C')
  else:
    tpu_num = sail.get_available_tpu_num()
    if args.n_worker < 0: args.n_worker = tpu_num
    print('>> TPU num:', tpu_num)
    print('>> n_worker:', args.n_worker)

    part = math.ceil(len(paths) / args.n_worker)
    lock = RLock()
    thrs = [
      Thread(target=worker, args=(i, args, paths[part*i:part*(i+1)], result, runtime, niqe, is_stop, lock), daemon=True) 
        for i in range(args.n_worker)
    ]

    try:
      for thr in thrs:
        thr.start()
      for thr in thrs:
        while True:
          thr.join(timeout=5)
          if not thr.is_alive():
            break
    except KeyboardInterrupt:
      is_stop.set()
      print('Exit by Ctrl+C')
    finally:
      thrs.clear()
  end_all = time()
  time_all = end_all - start_all
  print('time_all:', time_all)

  # gather results
  result.sort(key=(lambda e: e['img_name']))    # re-order
  metrics = {
    'A': [{
      'model_size': os.path.getsize(args.model), 
      'time_all': time_all, 
      'runtime_avg': format(mean(runtime), '.4f'),
      'niqe_avg': format(mean(niqe), '.4f'), 
      'images': result,
    }]
  }
  print('time_all:', metrics['A']['time_all'])
  print('runtime_avg:', metrics['A']['runtime_avg'])
  print('niqe_avg:', metrics['A']['niqe_avg'])

  print(f'>> saving to {args.report}')
  with open(args.report, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--device', type=int,  default=0,           help='TPU device id')
  parser.add_argument('-M', '--model',  type=Path, default=MODEL_FILE,  help='path to *.bmodel model ckpt')
  parser.add_argument('-I', '--input',  type=Path, default=IN_PATH,     help='input image or folder')
  parser.add_argument('-O', '--output', type=Path, default=IMAGE_PATH,  help='output image folder')
  parser.add_argument('-R', '--report', type=Path, default=REPORT_FILE, help='report model runtime to json file')
  parser.add_argument('-L', '--limit',  type=int,  default=-1, help='limit run sample count')
  parser.add_argument('--n_worker',     type=int,  default=0,  help='multi-thread workers')
  parser.add_argument('--save', action='store_true', help='save sr images')
  args = parser.parse_args()

  run(args)
