#!/usr/bin/env bash

# 样例模型 Real-ERSRGAN
pushd /workspace/code/repo/TPU-Coder-Cup/CCF2023/models

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

# list up files
ls
