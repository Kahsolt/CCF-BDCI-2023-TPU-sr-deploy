#!/usr/bin/env bash

# You should run this script in your docker container!

# go to the work dir
pushd /workspace

# unzip the tpu-mlir sdk
if [ ! -d "tpu-mlir" ]; then
  cp -v code/ref/tpu-mlir_v1.2.8-g32d7b3ec-20230802.tar.gz .
  tar zxvf tpu-mlir_v1.2.8-g32d7b3ec-20230802.tar.gz --strip-components=1 -C .
  mv tpu-mlir_v1.2.8-g32d7b3ec-20230802 tpu-mlir
fi

# install python3.7 (the fucking sdk is binded to this version)
# https://www.build-python-from-source.com/
PATH_HOME=/opt/python37
if [ ! -d $PATH_HOME ]; then
  # apt resource
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse"           >  /etc/apt/sources.list
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse"   >> /etc/apt/sources.list
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list
  echo "deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse"            >> /etc/apt/sources.list

  # build tools
  apt-get update
  apt-get install -y make build-essential wget curl llvm
  apt-get install -y libssl-dev zlib1g-dev libbz2-dev libreadline-dev xz-utils liblzma-dev

  # src tarball
  wget https://www.python.org/ftp/python/3.7.8/Python-3.7.8.tgz
  tar xzvf Python-3.7.8.tgz
  cd Python-3.7.8

  # compile & install
  ./configure --prefix=$PATH_HOME --enable-optimizations --with-lto --with-computed-gotos --with-system-ffi --enable-shared
  make -j "$(nproc)"
  #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/Python-3.7.8
  #./python -m test -j "$(nproc)"
  make altinstall

  # replace system default python3
  pushd /usr/bin
  rm python3
  ln -s /opt/python37/bin/python3.7 python3
  rm pip3
  ln -s /opt/python37/bin/pip3.7 pip3
  rm pydoc3
  ln -s /opt/python37/bin/pydoc3.7 pydoc3
  rm python3-config
  ln -s /opt/python37/bin/python3.7m-config python3-config
  popd

  export LD_LIBRARY_PATH=$PATH_HOME/lib:$LD_LIBRARY_PATH

  # pip resource
  python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

  # pip requirements for tpu-mlir
  pip3 install numpy pillow plotly opencv-python
  pip3 install torch torchvision
fi

# load env vars
if [ ! $TPU_MLIR_INIT ]; then
  source tpu-mlir/envsetup.sh
  export LD_LIBRARY_PATH=$PATH_HOME/lib:$LD_LIBRARY_PATH
  alias python=python3
  alias pip=pip3
  export TPU_MLIR_INIT=1
fi

# go back
popd
