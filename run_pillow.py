from threading import Thread, RLock
from argparse import ArgumentParser

from PIL.Image import Resampling

from run_utils import *

RESAMPLE_METHODS = {
  'original': None,
  'nearest':  Resampling.NEAREST,
  'lanczos':  Resampling.LANCZOS,
  'bilinear': Resampling.BILINEAR,
  'bicubic':  Resampling.BICUBIC,
  'box':      Resampling.BOX,
  'hamming':  Resampling.HAMMING,
}


def worker(thr_id:int, args, paths:List[Path], result:List[dict], runtime:List[float], niqe:List[float], lock:RLock):
  # setup model
  upscale_ratio = 4.0
  
  total = len(paths)
  for idx, fp in enumerate((tqdm if thr_id == 0 else list)(paths)):
    # 加载图片
    img_low = Image.open(fp).convert('RGB')
    W_tgt, H_tgt = [int(e*upscale_ratio) for e in img_low.size]

    # 上采样
    start = time()
    if args.method !='original':
      img_high = img_low.resize((W_tgt, H_tgt), RESAMPLE_METHODS[args.method])
    else:
      img_high = img_low
    end = time() - start
    with lock: runtime.append(end)

    # 后处理
    if args.postprocess:
      img_high = img_high.filter(getattr(ImageFilter, args.postprocess))

    # 保存图片
    if args.save:
      img_high.save(Path(args.output) / fp.name)

    # 计算niqe
    niqe_output = get_niqe(pil_to_np(img_high))

    with lock:
      niqe.append(niqe_output)
      result.append({'img_name': fp.stem, 'runtime': format(end, '.4f'), 'niqe': format(niqe_output, '.4f')})

    if thr_id == 0 and (idx + 1) % 10 == 0:
      print(f'>> [{idx+1}/{total}]: niqe {mean(niqe)}, time {mean(runtime)}')


def run(args):
  # in/out paths
  paths = fix_input_output_paths(args)

  # workers & task
  lock = RLock()
  part = math.ceil(len(paths) / args.n_worker)
  result:  List[dict]  = []
  runtime: List[float] = []
  niqe:    List[float] = []
  start_all = time()
  thrs = [
    Thread(target=worker, args=(i, args, paths[part*i:part*(i+1)], result, runtime, niqe, lock), daemon=True) 
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
    print('Exit by Ctrl+C')
  finally:
    thrs.clear()
  end_all = time()
  time_all = end_all - start_all
  runtime_avg = mean(runtime)
  niqe_avg = mean(niqe)
  print('time_all:', time_all)
  print('runtime_avg:', runtime_avg)
  print('niqe_avg:', niqe_avg)
  print('>> score:', get_score(niqe_avg, runtime_avg))

  # gather results
  if args.n_worker != 0:
    result.sort(key=(lambda e: e['img_name']))    # re-order
  ranklist = 'A' if args.dataset == 'test' else 'B'
  metrics = {
    ranklist: [{
      'method': args.method, 
      'time_all': time_all, 
      'runtime_avg': format(runtime_avg, '.4f'),
      'niqe_avg': format(niqe_avg, '.4f'), 
      'images': result,
    }]
  }
  print(f'>> saving to {args.report}')
  with open(args.report, 'w', encoding='utf-8') as fh:
    json.dump(metrics, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--method',  type=str,  default='lanczos', choices=RESAMPLE_METHODS.keys())
  parser.add_argument('-D', '--dataset', type=str,  default='val',     choices=DATASETS)
  parser.add_argument('-L', '--limit',   type=int,  default=-1,        help='limit dataset run sample count')
  parser.add_argument('--n_worker',      type=int,  default=-1,        help='multi-thread workers')
  parser.add_argument('-pp', '--postprocess', choices=POSTPROCESSOR)
  parser.add_argument('--save',          action='store_true',          help='save sr images')
  args = parser.parse_args()

  args.n_worker = min(args.n_worker, args.limit)
  if args.n_worker <= 0:
    cpu_num = os.cpu_count()
    args.n_worker = min(args.n_worker, cpu_num)
    print('>> CPU num:', cpu_num)
    print('>> n_worker:', args.n_worker)
  
  args.log_dp: Path = OUT_PATH / args.dataset / Path(args.method).stem
  args.log_dp.mkdir(parents=True, exist_ok=True)
  args.output = args.log_dp / 'test_sr'
  args.report = args.log_dp / 'test.json'

  run(args)
