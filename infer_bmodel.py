#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/11/16 

from upscale_bmodel import *


def run(args):
  # 加载模型
  upmodel = UpscaleModel(
    args.model, 
    model_size=(args.model_size, args.model_size), 
    upscale_rate=4, 
    tile_size=(args.tile_size, args.tile_size), 
    padding=args.padding, 
    device_id=0,
  )

  # 加载图片
  fp = Path(args.f)
  img = Image.open(fp)

  # 模型推理
  start = time()
  res = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0, ret_type='np')
  end = time() - start
  print('time:', end)

  # 保存图片
  if args.save:
    fp_out = Path(args.output) / fp.name
    Image.fromarray(res).save(fp_out)
    img = Image.open(fp_out)
  else:
    img = res

  # 计算niqe
  output = np.asarray(img)
  niqe_output = calculate_niqe(output, 0, input_order='HWC', convert_to='y')
  print('niqe:', niqe_output)


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('-f', type=Path, help='imaga path')
  args = get_args(parser)

  run(args)
