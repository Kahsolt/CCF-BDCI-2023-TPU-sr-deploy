#!/usr/bin/env bash

# 工作目录 (本项目挂载根目录)
pushd /workspace/code
# tpu_mlir sdk 环境
source setup_tpu_mlir.sh

# 编译模型保存目录
if [ ! -d models ]; then mkdir models; fi
cd models
if [ ! -d mobilenet_v2 ]; then mkdir mobilenet_v2; fi
cd mobilenet_v2

# 样例模型 MobileNetV2 (随 tpu-mlir v1.4 发布)
wget -nc https://github.com/sophgo/tpu-mlir/releases/download/v1.4-beta.0/mobilenet_v2.pt

# 将 torch.jit 模型转换为 mlir
model_transform.py \
 --model_name mobilenet_v2 \
 --input_shape [[1,3,200,200]] \
 --model_def mobilenet_v2.pt \
 --mlir mobilenet_v2.mlir

# 将 mlir 转换成 fp16 的 bmodel
model_deploy.py \
 --mlir mobilenet_v2.mlir \
 --quantize F16 \
 --chip bm1684x \
 --model mobilenet_v2.bmodel

# list up generated files
ls -l
