#!/usr/bin/env bash

if [ -z $1 ]; then
  echo "usage: $0 <model_subfolder_name> [channel_size]"
  exit -1
fi

MODEL_NAME=$1

if [ -z $2 ]; then
  C=3     # RGB models
else
  C=$2    # y_only models
fi

# 工作目录 (本项目挂载根目录)
pushd /workspace/code
# tpu_mlir sdk 环境
source setup_tpu_mlir.sh

# 编译模型保存目录
if [ ! -d models ]; then mkdir models; fi
cd models
if [ ! -d $MODEL_NAME ]; then
  echo ">> model subfolder not found, should run other init script first :("
  exit -1
fi
cd $MODEL_NAME

# TPU并不支持并行处理批
B=1
# 确定输入图片尺寸
H=192
W=256
# 设备
DEVICE=bm1684x
# 文件路径
MODEL_PATH=$MODEL_NAME.pt
MLIR_NAME=$MODEL_NAME.mlir
BMODEL_FP32_FILE=$MODEL_NAME.fp32.bmodel
BMODEL_FP16_FILE=$MODEL_NAME.fp16.bmodel

# 将 torch.jit 模型转换为 mlir
if [ ! -f $MLIR_NAME ]; then
  model_transform.py \
    --model_name $MODEL_NAME \
    --input_shape [[$B,$C,$H,$W]] \
    --model_def $MODEL_PATH \
    --mlir $MLIR_NAME
fi

# 将 mlir 转换成 fp32 的 bmodel
if [ ! -f $BMODEL_FP32_FILE ]; then
  model_deploy.py \
    --mlir $MLIR_NAME \
    --quantize F32 \
    --chip $DEVICE \
    --model $BMODEL_FP32_FILE
fi

# 将 mlir 转换成 fp16 的 bmodel
if [ ! -f $BMODEL_FP16_FILE ]; then
  model_deploy.py \
    --mlir $MLIR_NAME \
    --quantize F16 \
    --chip $DEVICE \
    --model $BMODEL_FP16_FILE
fi

# list up generated files
ls -l
