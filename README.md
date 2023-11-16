# CCF-BDCI-2023-TPU-sr-deploy

    CCF BDCI 2023 基于TPU平台实现超分辨率重建模型部署

----

比赛主页: [https://www.datafountain.cn/competitions/972](https://www.datafountain.cn/competitions/972)


### develop

⚪ use my prebuilt docker image

- install [Docker](https://docs.docker.com/get-docker/)
- run `run_docker.cmd`
- run following example in your docker container
  - `bash convert_resrgan.sh` (contest demo)
  - `bash convert_mobilenet_v2.sh` (sdk v1.4 demo)

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


#### references

- tpu-mlir
  - site: [https://tpumlir.org/](https://tpumlir.org/)
  - repo: [https://github.com/sophgo/tpu-mlir](https://github.com/sophgo/tpu-mlir)
  - doc: [https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html)
- sophnet tpu-cloud: [https://www.sophnet.com/](https://www.sophnet.com/)
- contest demo repo: [https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023](https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023)

----
by Armit
2023/11/14
