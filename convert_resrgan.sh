#!/usr/bin/env bash

# 工作目录 (本项目挂载根目录)
pushd /workspace/code
# tpu_mlir sdk 环境
source setup_tpu_mlir.sh

# 编译模型保存目录
if [ ! -d models ]; then mkdir models; fi
cd models
if [ ! -d r-esrgan ]; then mkdir r-esrgan; fi
cd r-esrgan

# 样例模型 Real-ERSRGAN (BDCI-2023 基于TPU平台实现超分辨率重建模型部署比赛)
wget -nc https://github.com/sophgo/TPU-Coder-Cup/raw/main/CCF2023/models/r-esrgan4x+.pt

# 将 torch.jit 模型转换为 mlir
model_transform.py \
 --model_name r-esrgan \
 --input_shape [[1,3,200,200]] \
 --model_def r-esrgan4x+.pt \
 --mlir r-esrgan4x.mlir

# 将 mlir 转换成 fp16 的 bmodel
model_deploy.py \
 --mlir r-esrgan4x.mlir \
 --quantize F16 \
 --chip bm1684x \
 --model r-esrgan4x.bmodel

# list up generated files
ls -l
