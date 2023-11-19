#!/usr/bin/env bash

export PATH=/opt/sophon/libsophon-current/bin:$PATH
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/opt/sophon/libsophon-0.4.6/data/:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/opt/sophon/sophon-ffmpeg_0.6.0/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/sophon/opencv-bmcpu_0.6.0/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/sophon/opencv-bmcpu_0.6.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/sophon/sophon-opencv_0.6.0/lib/:$LD_LIBRARY_PATH


if [ ! -d TPU-Coder-Cup ]; then
  git clone https://github.com/sophgo/TPU-Coder-Cup
fi
