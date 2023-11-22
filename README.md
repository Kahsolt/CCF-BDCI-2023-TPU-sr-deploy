# CCF-BDCI-2023-TPU-sr-deploy

    CCF BDCI 2023 基于TPU平台实现超分辨率重建模型部署

----

Contest page: [https://www.datafountain.cn/competitions/972](https://www.datafountain.cn/competitions/972)  
Team name: Absofastlutely  


### results

ℹ Upscale `x4` on various-sized images from `testA.zip` on sophon's [bm1684x](https://www.sophon.ai/product/introduce/bm1684x.html) TPU

⚪ FP16

| model | padding | filter | time | niqe | score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| original |    |        |        | 4.2733 |           |  |
| carn_m   | 16 |        | 0.9991 | 5.0776 |  277.5417 | baseline |
| carn     | 16 |        | 0.9605 | 5.0115 |  293.6364 | baseline |
| ninasr   | 16 |        | 0.7442 | 4.8958 |  389.8195 | baseline |
| espcn    | 16 | DETAIL | 0.6879 | 5.2197 |  387.9559 | y_only |
| espcn_nc | 16 | DETAIL | 0.5369 | 5.2365 |  494.6222 | y_only |
| espcn_nc |  0 |  EDGE  | 0.3915 | 5.9385 |  526.3948 | y_only |
| espcn_cp |  0 |  EDGE  | 0.3116 | 4.3396 | 1046.7720 |  |
| espcn_ex | 16 |        | 0.4264 | 5.8571 |  501.4783 |  |
| espcn_ex | 16 | DETAIL | 0.4206 | 5.3193 |  616.5087 |  |
| espcn_ex | -1 |        | 0.2301 | 5.4146 | 1094.2843 |  |
| espcn_ex |  0 | DETAIL | 0.1922 | 5.7573 | 1159.8901 |  |
| espcn_ex |  0 | DETAIL | 0.1910 | 5.2661 | 1378.9572 |  |
| espcn_ex |  0 |  EDGE  | 0.1924 | 4.4465 | 1661.0886 |  |
| espcn_ex |  0 | EDGE++ | 0.1910 | 4.4465 | 1673.4388 |  | 

⚪ FP32

⚠ F32 models is **much slower** than F16 ones, because `TFLOPS = 32(INT8) / 16(FP16) / 2(FP32)`

| model | padding | filter | time | niqe | score | comment |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| original |    |        |        | 4.2733 |           |  |
| fsrcnn   | 16 |        | 3.6149 | 4.9615 |   78.9931 | very slow |
| ninasr   | 16 |        | 1.0166 | 5.5107 |  240.0883 | y_only |
| espcn    | 16 |        | 0.7661 | 5.0328 |  366.1582 | y_only |
| espcn    | 16 | DETAIL | 0.7628 | 4.7576 |  392.6231 | y_only |
| espcn_nc | 16 | DETAIL | 0.6301 | 5.0394 |  444.4484 |  |
| espcn_ex | 16 |        | 0.6184 | 5.8532 |  346.3551 |  |


### develop

> compile a nice pretrained pytorch super-resolution model to TPU-supported bmodel

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
  - `source env/setup_tpu_mlir.sh`
  - `echo $PATH` should contain many mlir stuff
  - now you should be able to run the mlir tools
    - `ls $TPUC_ROOT/bin`
    - `ls $TPUC_ROOT/python/tools`
    - `ls $TPUC_ROOT/python/utils`
    - `ls $TPUC_ROOT/python/test`
    - `ls $TPUC_ROOT/python/samples`
  - verify by running the official demos
    - `cd test`
    - `bash env/convert_mobilenet_v2.sh` (sdk v1.4 demo)
    - `bash env/convert_resrgan.sh` (contest demo)
- freeze up an image `docker commit -p -a kahsolt -m "add tpu-mlir v1.4" tpu-mlir kahsolt/tpuc_dev:latest`

#### compile the target model as we submit

- run `python model_espcn.py` **out of the docker in Windows cmd** (if you natively use Linux, you need some modification on paths)
- run `bash convert_espcn.sh` **in the docker**, you will get the compiled `models/espcn*/espcn*.<dtyp>.bmodel`


### deploy

> eval the compiled bmodel on real TPU device, and gather the metrics result

- apply for a cloud server at [sophnet tpu-cloud](https://www.sophnet.com/)
  - open web terminal and `cd /tmp`, remeber this is your workdir
- setup runtime
  - upload script [env/setup_sophon.sh](setup_sophon.sh)
  - run `source setup_sophon.sh`, which activaties the envvars and clones the material repo
  - list up TPU devices by `bm-smi`, get the driver version like `0.4.8`
  - install the python package `sophon` with right version (find it in the repo `TPU-Coder-Cup/CCF2023/sophon-{version}-py3-none-any.whl`)
- deploy & eval the compiled bmodel
  - upload testset `testA.zip` and unzip inplace
  - upload bmodel `*.bmodel`
  - upload code `run_bmodel.py` and `run_utils.py`
  - run eval `python run_bmodel.py -M <name.bmodel>`
  - find the results at `out/<name>//test.json`


#### references

- bm1684x chip
  - page: [https://www.sophon.ai/product/introduce/bm1684x.html](https://www.sophon.ai/product/introduce/bm1684x.html)
  - doc: [https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/03/02/20/BM1684X%20Introduction%20V1.7.pdf]
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
