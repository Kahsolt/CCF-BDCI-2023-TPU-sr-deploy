#!/usr/bin/env bash

BUILD_PATH=build

MODEL_NAME=ninasr
MODEL_FILE=$MODEL_NAME.pt
MLIR_FILE=$MODEL_NAME.mlir
BMODEL_FILE=$MODEL_NAME.bmodel

pushd $BUILD_PATH

# 将 torch.jit 模型转换为 mlir
if [ ! -f $MLIR_FILE ]; then
  model_transform.py \
    --model_name $MODEL_NAME \
    --input_shape [[1,3,192,256]] \
    --model_def $MODEL_FILE \
    --mlir $MLIR_FILE
fi

# 将 mlir 转换成 fp16 的 bmodel
if [ ! -f $BMODEL_FILE ]; then
  model_deploy.py \
    --mlir $MLIR_FILE \
    --quantize F16 \
    --chip bm1684x \
    --model $BMODEL_FILE
fi

# list up generated files
ls -l

popd
