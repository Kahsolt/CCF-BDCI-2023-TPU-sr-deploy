# CCF-BDCI-2023-TPU-sr-deploy

    CCF BDCI 2023 基于TPU平台实现超分辨率重建模型部署

----

Contest page: [https://www.datafountain.cn/competitions/972](https://www.datafountain.cn/competitions/972)  
Team name: Absofastlutely  


### benchmark

ℹ Upscale `x4` with `padding = 16` on images from `testA.zip` (though `pad=4` is pratically enough for `r-esrgan`)
ℹ Test sample count is **160** for pytorch, **600** (all) for original & sail (bmodel)
ℹ Input dtype is `FP32`; shape is `(200, 200)` for `r-esrgan`, `(192, 256)` for other models

| backend | device | model | time | niqe | score |
| :-: | :-: | :-: | :-: | :-: | :-: |
|         | CPU | original  |         | 4.27326 |  |
| pytorch | GPU | r-esrgan  | 2.73058 | 3.93794 | 128.168529838589 |
| pytorch | GPU | carn      | 0.96182 | 5.59615 | 246.375452070004 |
| pytorch | GPU | carn_m    | 0.79665 | 5.75405 | 280.227360568978 |
| pytorch | GPU | edsr      | 0.79325 | 5.49612 | 309.190130993299 |
| pytorch | GPU | ninasr_b0 | 0.45504 | 5.47967 | 541.936112691118 |
| pytorch | GPU | ninasr_b1 | 0.84457 | 5.33448 | 305.611128201859 |

> time efficiency is much more important than quality metrics!
> now we focus on the `ninasr_b0` model :)


### develop

> compile a nice pretrained pytorch super-resolution model to TPU-supported bmodel

⚠ the `bm1684x` device only support `fp32` & `int8` :(

⚪ use my prebuilt docker image

- install [Docker](https://docs.docker.com/get-docker/)
- run `run_docker.cmd`
- run following example in your docker container
  - `bash convert_resrgan.sh` (contest demo)
  - `bash convert_mobilenet_v2.sh` (sdk v1.4 demo)
- now you can compile your any pytorch model to bmodel use this toolchain

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
    - `bash convert_mobilenet_v2.sh`
    - `bash convert_resrgan.sh`
- freeze up an image `docker commit -p -a kahsolt -m "add tpu-mlir v1.4" tpu-mlir kahsolt/tpuc_dev:latest`


### deploy

> run the compiled bmodel on real TPU device, and gather the metrics result

- apply for a cloud server at [sophnet tpu-cloud](https://www.sophnet.com/)
  - open web terminal and `cd /tmp`, remeber this is your workdir
- setup runtime
  - upload script [setup_sophon.sh](setup_sophon.sh)
  - run `source setup_sophon.sh`, which activaties the envvars and clones the material repo
  - list up TPU devices by `bm-smi`, get the driver version like `0.4.8`
  - install the python package `sophon` with right version (find it in the repo `TPU-Coder-Cup/CCF2023/sophon-{version}-py3-none-any.whl`)
- deploy & run the compiled bmodel
  - upload testset `testA.zip` and unzip inplace
  - upload your bmodel `*.bmodel`
  - upload script `upscale_bmodel.py`
  - run `python upscale_bmodel.py -M <name>`
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

----
by Armit
2023/11/14
