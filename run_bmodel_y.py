from PIL.Image import Resampling

from run_bmodel import *

# some models only process on the Y channel in YCrCb space


def run(args):
  # in/out paths
  if Path(args.input).is_file():
    paths = [Path(args.input)]
  else:
    paths = [Path(fp) for fp in sorted(glob.glob(os.path.join(str(args.input), '*')))]
  if args.limit > 0: paths = paths[:args.limit]
  if args.save: Path(args.output).mkdir(parents=True, exist_ok=True)

  # setup model
  upscale_factor = 4.0
  model = TiledSRModel(args.model, args.model_size, padding=args.padding, device_id=args.device)

  # workers & task
  start_all = time()
  total = len(paths)
  result:  List[dict]  = []
  runtime: List[float] = []
  niqe:    List[float] = []
  for idx, fp in enumerate(tqdm(paths)):
    # 加载图片
    img = Image.open(fp).convert('RGB')
    im_low = pil_to_np(img)

    # 模型推理
    start = time()
    im_y, im_cb, im_cr = get_y_cb_cr(im_low)
    w_tgt, h_tgt = int(img.width * upscale_factor), int(img.height * upscale_factor)
    im_cb_high = pil_to_np(np_to_pil(im_cb).resize((w_tgt, h_tgt), Resampling.BICUBIC))
    im_cr_high = pil_to_np(np_to_pil(im_cr).resize((w_tgt, h_tgt), Resampling.BICUBIC))
    # [H, W, C=1], float32
    im_y = np.expand_dims(im_y, axis=-1)
    im_y_high = model(im_y)
    im_y_high = im_y_high.squeeze(-1)
    im_high_ycbcr = np.stack([im_y_high, im_cb_high, im_cr_high], axis=-1)
    im_high_bgr = ycbcr_to_bgr(im_high_ycbcr)
    im_high = bgr2rgb(im_high_bgr)
    end = time() - start
    runtime.append(end)

    img_high = None

    # 后处理
    if args.postprocess:
      img_high = img_high or np_to_pil(im_high)
      img_high = img_high.filter(ImageFilter.DETAIL)
      im_high = pil_to_np(img_high)

    # 保存图片
    if args.save:
      img_high = img_high or np_to_pil(im_high)
      img_high.save(Path(args.output) / fp.name)

    # 计算niqe
    niqe_output = get_niqe(im_high)
    niqe.append(niqe_output)

    result.append({'img_name': fp.stem, 'runtime': format(end, '.4f'), 'niqe': format(niqe_output, '.4f')})

    if (idx + 1) % 10 == 0:
      print(f'>> [{idx+1}/{total}]: niqe {mean(niqe)}, time {mean(runtime)}')

  end_all = time()
  time_all = end_all - start_all
  runtime_avg = mean(runtime)
  niqe_avg = mean(niqe)
  print('time_all:',    time_all)
  print('runtime_avg:', runtime_avg)
  print('niqe_avg:',    niqe_avg)
  print('>> score:',    get_score(niqe_avg, runtime_avg))

  # gather results
  metrics = {
    'A': [{
      'model_size': os.path.getsize(args.model), 
      'time_all': time_all, 
      'runtime_avg': format(mean(runtime), '.4f'),
      'niqe_avg': format(mean(niqe), '.4f'), 
      'images': result,
    }]
  }
  print(f'>> saving to {args.report}')
  with open(args.report, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  run(get_args())
