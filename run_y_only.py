from PIL.Image import Resampling

from run_utils import *

# some models only process on the Y channel in YCrCb space


def process_images(args, model:Callable, paths:List[Path], niqe:List[float], runtime:List[float], result:List[dict]):
  upscale_factor = 4.0
  total = len(paths)
  for idx, fp in enumerate(tqdm(paths)):
    # 加载图片
    img = Image.open(fp).convert('RGB')
    im_low = pil_to_np(img)

    # 模型推理
    start = time()
    im_y, im_cb, im_cr = rgb_to_y_cb_cr(im_low)
    w_tgt, h_tgt = int(img.width * upscale_factor), int(img.height * upscale_factor)
    im_cb_high = pil_to_np(np_to_pil(im_cb).resize((w_tgt, h_tgt), Resampling.BICUBIC))
    im_cr_high = pil_to_np(np_to_pil(im_cr).resize((w_tgt, h_tgt), Resampling.BICUBIC))
    # [H, W, C=1], float32
    im_y = np.expand_dims(im_y, axis=-1)
    im_y_high: ndarray = model(im_y)   # NOTE: vrng might be not normalized
    im_y_high = im_y_high.squeeze(-1)
    im_high_ycbcr = np.stack([im_y_high, im_cb_high, im_cr_high], axis=-1)
    im_high = ycbcr_to_rgb(im_high_ycbcr)
    end = time() - start
    runtime.append(end)

    im_high = im_high.clip(0.0, 1.0)    # vrng 0~1
    img_high = None

    # 后处理
    if args.postprocess:
      img_high = img_high or np_to_pil(im_high)
      img_high = img_high.filter(getattr(ImageFilter, args.postprocess))
      im_high = pil_to_np(img_high)

    # 保存图片
    if args.save:
      img_high = img_high or np_to_pil(im_high)
      img_high.save(Path(args.output) / fp.name)

    # 计算niqe
    niqe_output = get_niqe_y(im_y_high.clip(0.0, 1.0))  # vrng 0~1
    niqe.append(niqe_output)

    result.append({'img_name': fp.stem, 'runtime': format(end, '.4f'), 'niqe': format(niqe_output, '.4f')})

    if (idx + 1) % 10 == 0:
      print(f'>> [{idx+1}/{total}]: niqe {mean(niqe)}, time {mean(runtime)}')


if __name__ == '__main__':
  args = get_args()
  args = process_args(args)

  print(f'>> backend: {args.backend}')
  if args.backend == 'pytorch':
    from run_pytorch import get_model
  elif args.backend == 'bmodel':
    from run_bmodel import get_model
  run_eval(args, get_model, process_images)
