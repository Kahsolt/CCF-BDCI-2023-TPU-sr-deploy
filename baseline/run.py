#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/17 

import warnings ; warnings.simplefilter('ignore', category=RuntimeWarning)

import os
import math
import glob
import json
import time
import random
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, List

from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as PILImage
from numpy import ndarray
import numpy as np

from metrics.niqe import calculate_niqe
import sophon.sail as sail
sail.set_print_flag(False)
sail.set_dump_io_flag(False)

random.seed(42)
np.random.seed(42)

Box = Tuple[slice, slice]

mean = lambda x: sum([float(e) for e in x]) / len(x) if len(x) else 0.0
get_score = lambda niqe_avg, runtime_avg: math.sqrt(7 - niqe_avg) / runtime_avg * 200

def pil_to_np(img:PILImage) -> ndarray:
  return np.asarray(img, dtype=np.float32) / 255.0

def np_to_pil(im:ndarray) -> PILImage:
  return Image.fromarray(np.asarray(im * 255).astype(np.uint8))


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


class TiledSRModel:

  def __init__(self, model_fp:Path, model_size:Tuple[int, int]=(192, 256), padding=16, device_id=0):
    print(f'>> load model: {model_fp.stem}')
    self.model = EngineOV(str(model_fp), device_id)
    self.upscale_rate = 4.0
    self.tile_size = model_size  # (h, w)
    self.padding = padding

  @property
  def tile_h(self): return self.tile_size[0]
  @property
  def tile_w(self): return self.tile_size[1]

  def __call__(self, im:ndarray) -> ndarray:
    # [H, W, C=3]
    H, W, C = im.shape
    H_tgt, W_tgt = int(H * self.upscale_rate), int(W * self.upscale_rate)
    # tile count along aixs
    num_rows = math.ceil((H - self.padding) / (self.tile_h - self.padding))
    num_cols = math.ceil((W - self.padding) / (self.tile_w - self.padding))
    # uncrop (zero padding)
    H_ex = num_rows * self.tile_h - ((num_rows - 1) * self.padding)
    W_ex = num_cols * self.tile_w - ((num_cols - 1) * self.padding)
    im_ex = np.zeros([H_ex, W_ex, C], dtype=im.dtype)
    # relocate top-left origin
    init_y = (H_ex - H) // 2
    init_x = (W_ex - W) // 2
    # paste original image in the center
    im_ex[init_y:init_y+H, init_x:init_x+W, :] = im

    # [B=1, C=3, H_ex, W_ex]
    X = np.expand_dims(np.transpose(im_ex, (2, 0, 1)), axis=0)

    # break up tiles
    boxes_low:  List[Box] = []
    boxes_high: List[Box] = []
    y = 0
    while y + self.padding < H_ex:
      x = 0
      while x + self.padding < W_ex:
        boxes_low.append((
          slice(y, y + self.tile_h), 
          slice(x, x + self.tile_w),
        ))
        boxes_high.append((
          slice(int(y * self.upscale_rate), int((y + self.tile_h) * self.upscale_rate)), 
          slice(int(x * self.upscale_rate), int((x + self.tile_w) * self.upscale_rate)),
        ))
        x += self.tile_w - self.padding
      y += self.tile_h - self.padding
    n_tiles = len(boxes_low)
    assert n_tiles == num_rows * num_cols

    # forward & sew up tiles
    H_ex_tgt, W_ex_tgt = int(H_ex * self.upscale_rate), int(W_ex * self.upscale_rate)
    canvas = np.zeros([C, H_ex_tgt, W_ex_tgt], dtype=X.dtype)
    count  = np.zeros([   H_ex_tgt, W_ex_tgt], dtype=np.int32)
    for i in range(len(boxes_low)):
      low_slices  = boxes_low [i]
      high_slices = boxes_high[i]
      # [B=1, C, H_tile=192, W_tile=256]
      low_h, low_w = low_slices
      XT = X[:, :, low_h, low_w]
      # [B=1, C, H_tile*F=764, W_tile*F=1024]
      YT: ndarray = self.model([XT])[0][0]
      # paste to canvas
      high_h, high_w = high_slices
      count [   high_h, high_w] += 1
      canvas[:, high_h, high_w] += YT

    # handle overlap
    out_ex = np.where(count > 1, canvas / count, canvas)
    # crop
    fin_y = int(init_y * self.upscale_rate)
    fin_x = int(init_x * self.upscale_rate)
    out = out_ex[:, fin_y:fin_y+H_tgt, fin_x:fin_x+W_tgt]
    # vrng, to HWC
    out = np.transpose(out, [1, 2, 0])
    # numpy & clip
    return out.clip(0.0, 1.0).astype(np.float32)


def run(args):
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, "*")))
    if args.limit: paths = paths[:args.limit]

    model = TiledSRModel(args.model_path)

    start_all = time.time()
    total = len(paths)
    result, runtime, niqe = [], [], []
    for idx, path in enumerate(tqdm(paths)):
        img_name, extension = os.path.splitext(os.path.basename(path))
        img = Image.open(path).convert('RGB')
        im_low = pil_to_np(img)

        start = time.time()
        im_high = model(im_low)
        end = format((time.time() - start), '.4f')
        runtime.append(end)

        if not args.no_save:
            output_path = os.path.join(args.output, img_name + extension)
            np_to_pil(im_high).save(output_path)

        # 计算niqe
        out_put = im_high[:, :, ::-1]  # rgb2bgr
        out_put = np.asarray(out_put * 255, dtype=np.uint8)
        niqe_output = calculate_niqe(out_put, 0, input_order='HWC', convert_to='y')
        niqe_output = format(niqe_output, '.4f')
        niqe.append(niqe_output)

        result.append({"img_name": img_name, "runtime": end, "niqe": niqe_output})

        if (idx + 1) % 10 == 0:
            running_niqe = mean(niqe)
            running_time = mean(runtime)
            try:    running_score = get_score(running_niqe, running_time)
            except: running_score = 'NaN'
            print(f'>> [{idx+1}/{total}]: niqe {running_niqe}, time {running_time}, score {running_score}')

    model_size = os.path.getsize(args.model_path)
    runtime_avg = mean(runtime)
    niqe_avg = mean(niqe)

    end_all = time.time()
    time_all = end_all - start_all
    print('time_all:', time_all)
    params = {"A": [{
        "model_size": model_size, 
        "time_all": time_all, 
        "runtime_avg": format(runtime_avg, '.4f'),
        "niqe_avg": format(niqe_avg, '.4f'), 
        "images": result,
    }]}
    print("runtime_avg: ", runtime_avg)
    print("niqe_avg: ", niqe_avg)
    try:    print("score: ", get_score(niqe_avg, runtime_avg))
    except: print("score: NaN")

    output_fn = f'{args.report}'
    with open(output_fn, 'w') as f:
        json.dump(params, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=Path, default="./ninasr.bmodel",     help="Model names")
    parser.add_argument("-i", "--input",      type=Path, default="./dataset/test",      help="Input image or folder")
    parser.add_argument("-o", "--output",     type=Path, default="./results/test_fix",  help="Output image folder")
    parser.add_argument("-r", "--report",     type=Path, default="./results/test.json", help="report model runtime to json file")
    parser.add_argument("-l", "--limit",      type=int)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    assert os.path.isfile(args.model_path)
    assert os.path.isdir(args.input)
    if not args.no_save: os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    run(args)
