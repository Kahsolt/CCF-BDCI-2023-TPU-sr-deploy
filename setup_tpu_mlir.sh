#!/usr/bin/env bash

# You should run this script in your docker container!

# go to the work dir
pushd /workspace

# tpu-mlir sdk & system init
if [ ! -d "tpu-mlir" ]; then
  # unzip the tpu-mlir sdk
  wget -nc https://github.com/sophgo/tpu-mlir/releases/download/v1.4-beta.0/tpu-mlir_v1.4.beta.0-20230927.tar.gz
  tar zxvf tpu-mlir_v1.4.beta.0-20230927.tar.gz --strip-components=1 -C .
  mv tpu-mlir_v1.4.beta.0-20230927 tpu-mlir
  rm -f tpu-mlir_v1.4.beta.0-20230927.tar.gz

  # apt resource
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse"           >  /etc/apt/sources.list
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse"   >> /etc/apt/sources.list
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list
  echo "deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse"            >> /etc/apt/sources.list

  # pip resource
  python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
fi

# load env vars
if [ ! $TPU_MLIR_INIT ]; then
  source tpu-mlir/envsetup.sh
  export LD_LIBRARY_PATH=$PATH_HOME/lib:$LD_LIBRARY_PATH
  alias python=python3
  alias py=python3
  alias pip=pip3
  export TPU_MLIR_INIT=1
fi

# go back to workdir
popd
