from threading import Thread

from run_y_only import *


def task_cbcr(im_cb:ndarray, im_cr:ndarray, img:PILImage, cbcr_high:List):
  ts_cbcr = time()
  upscale_factor = 4.0
  w_tgt, h_tgt = int(img.width * upscale_factor), int(img.height * upscale_factor)
  im_cb_high = pil_to_np(np_to_pil(im_cb).resize((w_tgt, h_tgt), Resampling.BICUBIC))
  im_cr_high = pil_to_np(np_to_pil(im_cr).resize((w_tgt, h_tgt), Resampling.BICUBIC))
  cbcr_high.extend([
    im_cb_high,
    im_cr_high,
  ])
  if DEBUG_TIME: print('ts_cbcr:', time() - ts_cbcr)
  
def task_y(im_y:ndarray, model, ret:List):
  ts_y = time()
  im_y = np.expand_dims(im_y, axis=-1)
  im_y_high: ndarray = model(im_y)    # NOTE: vrng might be not normalized
  im_y_high = im_y_high.squeeze(-1)
  ret.append(im_y_high)
  if DEBUG_TIME: print('ts_y:', time() - ts_y)


def process_images(args, model:Callable, paths:List[Path], niqe:List[float], runtime:List[float], result:List[dict]):
  total = len(paths)
  for idx, fp in enumerate(tqdm(paths)):
    # 加载图片
    img = Image.open(fp).convert('RGB')
    im_low = pil_to_np(img)

    # 模型推理
    start = time()
    im_y, im_cb, im_cr = rgb_to_y_cb_cr(im_low)
    if DEBUG_TIME: print('ts_rgb_to_y_cb_cr:', time() - start)

    cbcr_high = []
    y_high = []
    thrs = [
      Thread(target=task_cbcr, args=(im_cb, im_cr, img, cbcr_high), daemon=True),
      Thread(target=task_y,    args=(im_y, model, y_high),          daemon=True),
    ]
    for thr in thrs: thr.start()
    for thr in thrs: thr.join()

    im_cb_high, im_cr_high = cbcr_high
    im_y_high = y_high[0]

    ts_stack = time()
    im_high_ycbcr = np.stack([im_y_high, im_cb_high, im_cr_high], axis=-1)
    if DEBUG_TIME: print('ts_stack:', time() - ts_stack)

    ts_ycbcr_to_rgb = time()
    im_high = ycbcr_to_rgb(im_high_ycbcr)
    if DEBUG_TIME: print('ts_ycbcr_to_rgb:', time() - ts_ycbcr_to_rgb)

    end = time() - start
    runtime.append(end)
    if DEBUG_TIME: print('ts_split_infer_combine:', end)

    im_high = im_high.clip(0.0, 1.0)    # vrng 0~1
    img_high = None

    # 后处理
    if args.postprocess:
      if DEBUG_TIME: ts_pp = time()
      img_high = img_high or np_to_pil(im_high)
      img_high = img_high.filter(getattr(ImageFilter, args.postprocess))
      im_high = pil_to_np(img_high)
      if DEBUG_TIME: print('ts_pp:', time() - ts_pp)

    # 保存图片
    if args.save:
      if DEBUG_TIME: ts_save = time()
      img_high = img_high or np_to_pil(im_high)
      img_high.save(Path(args.output) / fp.name)
      if DEBUG_TIME: print('ts_save:', time() - ts_save)

    # 计算niqe
    if DEBUG_TIME: ts_niqe = time()
    niqe_output = get_niqe_y(im_y_high.clip(0.0, 1.0))  # vrng 0~1
    if DEBUG_TIME: print('ts_niqe:', time() - ts_niqe)
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
