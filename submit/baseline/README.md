# CCF-BDCI-2023-TPU-sr-deploy baseline

    This folder contains code for the extra **baseline recruit award** :)

----

=> link: [https://discussion.datafountain.cn/articles/detail/3877](https://discussion.datafountain.cn/articles/detail/3877)


### model

We provide the `carn`, `carn_m` and `ninasr_b0_x4` solution as reference baselines.  
[NinaSR](https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md) is a self-designed light-weighted image super-resolution network.  
It processes full RGB channels end-to-end :)  

⚠ The `ninasr_b0_x4` model does not yield seamless output, still needs a lot of performance improvments. :(  

### run

⚪ run in your docker develop env

- following example scripts works for `ninasr_b0_x4`
- `python convert.py` load weights and convert to torch.jit script_module
- `bash compile.sh` compile script_module => mlir => bmodel using tpu-milr toolchain

⚪ run in the sophon-clound deploy env

- upload files
    - dataset `testA.zip`, unzip to `/tmp/dataset/test`
    - bmodel `build/ninasr.bmodel` to `/tmp`
    - code `run.py` and `metrics` to `/tmp`
- run infer `run.py`, or `run.py --no_save` if you're not saving output images
- find results at `./results`

### results

> score = sqrt(7 - niqe_avg) / time_avg * 200
> original dataset niqe: 4.27326

⚠ 我记不得下面这个结果是 `carn`, `carn_m`, `ninasr_b0_x4` 中的哪个模型跑出来的了，您可以自己都试试……

| time_avg | niqe_avg | score |
| :-: | :-: | :-: |
| 1.0166 | 5.5107 | 240.0883 |

----
by Armit
2023/11/18
