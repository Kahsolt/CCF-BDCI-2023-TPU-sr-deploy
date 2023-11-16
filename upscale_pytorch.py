import sys
import os
import glob
import math
import json
from time import time
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage
from argparse import ArgumentParser
import warnings

import torch
from torch import Tensor
from torch.nn import Module

BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / 'models'
MODEL_FILE = MODEL_PATH / 'r-esrgan' / 'r-esrgan4x+.pt'
LIB_PATH = BASE_PATH / 'repo' / 'TPU-Coder-Cup' / 'CCF2023'
IN_PATH = LIB_PATH / 'dataset' / 'test'
OUT_PATH = BASE_PATH / 'out' ; OUT_PATH.mkdir(exist_ok=True)

if not MODEL_FILE.is_file():
  MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
  cwd = os.getcwd()
  os.chdir(MODEL_FILE.parent)
  os.system('wget -nc https://github.com/sophgo/TPU-Coder-Cup/raw/main/CCF2023/models/r-esrgan4x+.pt')
  os.chdir(cwd)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the contest scaffold
sys.path.append(str(LIB_PATH))
from fix import *
from metrics.niqe import calculate_niqe


# ref: https://github.com/sophgo/TPU-Coder-Cup/blob/main/CCF2023/upscale.py
class UpscaleModel:

  def __init__(self, model_fp, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=4):
    self.model: Module = torch.load(model_fp, map_location='cpu')
    self.model = self.model.eval().to(device)
    self.dtype = list(self.model.parameters())[0].dtype
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

  @torch.inference_mode()
  def model_process(self, tile:PILImage):
    # preprocess
    ntile = tile.resize(self.model_size)  # (216, 216) => (200, 200)
    ntile = np.asarray(ntile).astype(np.float32)
    ntile = ntile / 255
    ntile = np.transpose(ntile, (2, 0, 1))
    ntile = ntile[np.newaxis, :, :, :]    # [B=1, C=3, H=200, W=200]
    # model forward
    X = torch.from_numpy(ntile).to(device=device, dtype=self.dtype)
    out: Tensor = self.model(X)           # [B=1, C=3, H=800, W=800]
    out = out.clamp_(0.0, 1.0)
    res = out[0].cpu().numpy()            # [C=3, H=800, W=800]
    # extract padding
    res = np.transpose(res, (1, 2, 0))
    res = res * 255
    res = res.astype(np.uint8)
    res = Image.fromarray(res)
    res = res.resize(self.target_tile_size)   # (800, 800) => (864, 864)
    return res

  def extract_and_enhance_tiles(self, image:PILImage, upscale_ratio:float=2.0):
    if image.mode != 'RGB':
      image = image.convert('RGB')
    # 获取图像的宽度和高度
    width, height = image.size
    self.upscale_rate = upscale_ratio
    self.target_tile_size = (int((self.tile_size[0] + self.padding * 1) * upscale_ratio),
                 int((self.tile_size[1] + self.padding * 1) * upscale_ratio))
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


def run(args):
  # in/out paths
  if Path(args.input).is_file():
    paths = [Path(args.input)]
  else:
    paths = [Path(fp) for fp in sorted(glob.glob(os.path.join(str(args.input), '*')))]
  Path(args.output).mkdir(parents=True, exist_ok=True)

  # set models
  upmodel = UpscaleModel(args.model_path, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=20)

  start_all = time()
  result, runtime, niqe = [], [], []
  for idx, fp in enumerate(paths):
    # 加载图片
    print(f'Testing {idx}: {fp.name}')
    img = Image.open(fp)

    # 模型推理
    start = time()
    res = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0)
    end = format((time() - start), '.4f')
    runtime.append(end)

    # 保存图片
    fp_out = Path(args.output) / fp.name
    res.save(fp_out)

    # 计算niqe
    img = Image.open(fp_out)
    output = np.asarray(img)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', category=RuntimeWarning)
      niqe_output = calculate_niqe(output, 0, input_order='HWC', convert_to='y')
    niqe_output = format(niqe_output, '.4f')
    niqe.append(niqe_output)

    result.append({'img_name': fp.stem, 'runtime': end, 'niqe': niqe_output})

  model_file_size = os.path.getsize(args.model_path)
  runtime_avg = np.mean(np.asarray(runtime, dtype=float))
  niqe_avg = np.mean(np.asarray(niqe, dtype=float))

  end_all = time()
  time_all = end_all - start_all
  print('time_all:', time_all)
  metrics = {
    'A': [{
      'model_size': model_file_size, 
      'time_all': time_all, 
      'runtime_avg': format(runtime_avg, '.4f'),
      'niqe_avg': format(niqe_avg, '.4f'), 
      'images': result,
    }]
  }
  print('metrics:', metrics)

  with open(args.report, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model_path', type=Path, default=MODEL_FILE,             help='path to *.pt model ckpt')
  parser.add_argument('-I', '--input',      type=Path, default=IN_PATH,                help='input image or folder')
  parser.add_argument('-O', '--output',     type=Path, default=OUT_PATH / 'test_sr',   help='output image folder')
  parser.add_argument('-R', '--report',     type=Path, default=OUT_PATH / 'test.json', help='report model runtime to json file')
  args = parser.parse_args()

  run(args)