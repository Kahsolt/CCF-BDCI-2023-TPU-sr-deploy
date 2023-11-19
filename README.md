# CCF-BDCI-2023-TPU-sr-deploy

    CCF BDCI 2023 基于TPU平台实现超分辨率重建模型部署

----

Contest page: [https://www.datafountain.cn/competitions/972](https://www.datafountain.cn/competitions/972)  
Team name: Absofastlutely  


### benchmark

⚪ pytorch inference (GPU)

ℹ Upscale `x4` with `padding = 16` on images from `testA.zip` (though `pad=4` is pratically enough for `r-esrgan`)
ℹ Test sample count is **160** for pytorch, **600** (all) for original & sail (bmodel)
ℹ Input dtype is `FP32`; shape is `(200, 200)` for `r-esrgan`, `(192, 256)` for other models

| model | time | niqe | score |
| :-: | :-: | :-: | :-: |
| original  |         | 4.27326 |  |
| r-esrgan  | 2.73058 | 3.93794 | 128.168529838589 |
| carn      | 0.96182 | 5.59615 | 246.375452070004 |
| carn_m    | 0.79665 | 5.75405 | 280.227360568978 |
| edsr      | 0.79325 | 5.49612 | 309.190130993299 |
| ninasr_b0 | 0.45504 | 5.47967 | 541.936112691118 |
| ninasr_b1 | 0.84457 | 5.33448 | 305.611128201859 |

> time efficiency is much more important than quality metrics!
> hence we'll focus on migrating the tiny models to TPU :)

⚪ bmodel inference (TPU)

| model | dtype | time | niqe | score |
| :-: | :-: | :-: | :-: | :-: |
| ninasr   | FP16 | 0.7442 | 4.8958 | 389.8195 |
| ninasr   | FP32 | 1.0166 | 5.5107 | 240.0883 |
| carn_m   | FP16 | 0.9991 | 5.0776 | 277.5417 |
| carn     | FP16 | 0.9605 | 5.0115 | 293.6364 |
| fsrcnn   | FP32 | 3.6149 | 4.9615 |  78.9931 |
| espcn    | FP32 | 0.7661 | 5.0328 | 366.1582 |
| espcn-pp | FP32 | 0.7628 | 4.7576 | 392.6231 |

⚪ dummy bmodel inference (TPU)

> Get to know the real TPU computational capacity :(
> The theoretical score upper limit can be estimated: sqrt(7-4)/0.53\*200 ≈ 650
> The max score for ESPCN-based models should be about: sqrt(7-4.5)/0.75\*200 ≈ 420

ℹ Set test examples `--limit 16`, the computation is not enough complex to show `INT8` advantage (?); however, it shows the basic overhead on TPU

| model | dtype | time |
| :-: | :-: | :-: |
| empty | FP32 | 0.52173 |
| empty | INT8 | 0.52447 |
| cheap | FP32 | 0.54948 |
| cheap | INT8 | 0.53010 |


### develop

> compile a nice pretrained pytorch super-resolution model to TPU-supported bmodel

⚠ the `bm1684x` device only support `fp32` & `int8` :(

#### launch the docker dev env

⚪ use my prebuilt docker image

- install [Docker](https://docs.docker.com/get-docker/)
- run `run_docker.cmd` to start the docker conatiner
  - now you can compile your any pytorch model to bmodel use the `tpu-mlir` toolchain

⚪ build by yourself

⚠ What will you do: add [tpu-mlir sdk](https://github.com/sophgo/tpu-mlir) to the official dev-env docker image `sophgo/tpuc_dev`  
⚠ Follow me if you're on **Windows**, otherwise refer to the [official tutorial](https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023#13-%E9%85%8D%E7%BD%AE%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83) for **Linux** systems  

> NOTE: the official tutorial is much out-dated, it binds `tpu-mlir_v1.2` to `sophgo/tpuc_dev:v2.2`, which works for Python3.7; however, we already have `tpu-mlir_v1.5` and `sophgo/tpuc_dev:v3.2` working with Python3.10 now :(

- install [Docker](https://docs.docker.com/get-docker/)
- `docker pull sophgo/tpuc_dev:latest` (see like [ChatGLM3-TPU](https://github.com/sophgo/ChatGLM3-TPU) or [Llama2-TPU](https://github.com/sophgo/Llama2-TPU))
- `CD /D /path/to/this/repo`
- `docker run --name tpu-mlir --volume="%CD%":/workspace/code --workdir /workspace/code -it -d sophgo/tpuc_dev:latest`
- setup the docker container (run the following Linux shell commands in your container)
  - `source setup_tpu_mlir.sh`
  - `echo $PATH` should contain many mlir stuff
  - now you should be able to run the mlir tools
    - `ls $TPUC_ROOT/bin`
    - `ls $TPUC_ROOT/python/tools`
    - `ls $TPUC_ROOT/python/utils`
    - `ls $TPUC_ROOT/python/test`
    - `ls $TPUC_ROOT/python/samples`
  - verify by running the official demos
    - `bash convert_mobilenet_v2.sh` (sdk v1.4 demo)
    - `bash convert_resrgan.sh` (contest demo)
- freeze up an image `docker commit -p -a kahsolt -m "add tpu-mlir v1.4" tpu-mlir kahsolt/tpuc_dev:latest`

#### compile the target model as we submit

- run `bash convert_epscn.sh` in the docker
- you will get `models/espcn/espcn.<dtyp>.bmodel`


### deploy

> eval the compiled bmodel on real TPU device, and gather the metrics result

- apply for a cloud server at [sophnet tpu-cloud](https://www.sophnet.com/)
  - open web terminal and `cd /tmp`, remeber this is your workdir
- setup runtime
  - upload script [setup_sophon.sh](setup_sophon.sh)
  - run `source setup_sophon.sh`, which activaties the envvars and clones the material repo
  - list up TPU devices by `bm-smi`, get the driver version like `0.4.8`
  - install the python package `sophon` with right version (find it in the repo `TPU-Coder-Cup/CCF2023/sophon-{version}-py3-none-any.whl`)
- deploy & eval the compiled bmodel
  - upload testset `testA.zip` and unzip inplace
  - upload bmodel `*.bmodel`
  - upload code `run_y_only.py`, `run_bmodel.py` and `run_utils.py`
  - run `python run_y_only.py -M <name>`
  - find the results at `out/test.json`


#### references

- tpu-mlir
  - site: [https://tpumlir.org/](https://tpumlir.org/)
  - repo: [https://github.com/sophgo/tpu-mlir](https://github.com/sophgo/tpu-mlir)
  - doc: [https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html)
- sophnet tpu-cloud: [https://www.sophnet.com/](https://www.sophnet.com/)
- contest demo repo: [https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023](https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023)
- torchSR: [https://github.com/Coloquinte/torchSR](https://github.com/Coloquinte/torchSR)
 - NinaSR: [https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md](https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md)
- ESPCN-PyTorch: [https://github.com/Lornatang/ESPCN-PyTorch](https://github.com/Lornatang/ESPCN-PyTorch)
  - @Lornatang: [https://github.com/Lornatang](https://github.com/Lornatang)
  - model weight zoo: [https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

----
by Armit
2023/11/14
