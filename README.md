# CCF-BDCI-2023-TPU-sr-deploy

    CCF BDCI 2023 基于TPU平台实现超分辨率重建模型部署

----

比赛主页: [https://www.datafountain.cn/competitions/972](https://www.datafountain.cn/competitions/972)


### develop

- install [Docker](https://docs.docker.com/get-docker/)
- run `repo\init_repos.cmd`

⚪ use my prebuilt docker image

- `CD /D /path/to/this/repo`
- `docker run --name tpu-mlir --volume="%CD%":/workspace/code --workdir /workspace/code -it -d kahsolt/tpuc_dev-py37:latest`

⚪ build from scratch by yourself

**WHY YOU NEED THIS: the official `sophgo/tpuc_dev:latest` offers py310, but the given tpu-mlir sdk is binded to py37, what the fuck :(**

⚠ Follow me if you're on **Windows**, otherwise follow the [official tutorial](https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023#13-%E9%85%8D%E7%BD%AE%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83) for **Linux** systems

- `docker pull sophgo/tpuc_dev:latest`
  - like in [ChatGLM3-TPU](https://github.com/sophgo/ChatGLM3-TPU) or [Llama2-TPU](https://github.com/sophgo/Llama2-TPU)
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
- pack an image `docker commit -p -a kahsolt -m "add python3.7 & tpu-mlir sdk" tpu-mlir kahsolt/tpuc_dev-py37:latest`

⚪ verify your docker image setup installations

- activate the envvars `source setup_tpu_mlir.sh`
- run the official demo `bash convert_resrgan.sh`


#### references

- tpu-mlir
  - site: [https://tpumlir.org/](https://tpumlir.org/)
  - doc: [https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html)
  - demo repo: [https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023](https://github.com/sophgo/TPU-Coder-Cup/tree/main/CCF2023)
- sophnet tpu-cloud: [https://www.sophnet.com/](https://www.sophnet.com/)

----
by Armit
2023/11/14
