#!/usr/bin/env bash

if [ -z $1 ]; then
  echo "usage: $0 <model_subfolder_name> [B] [C] [H] [W]"
  echo "example:"
  echo "  bash ./convert.sh ninasr_b0_x4"
  echo "  bash ./convert.sh espcn_nc 1 1 192 256"
  echo "  bash ./convert.sh espcn_ex 1 3 512 512"
  exit -1
fi

MODEL_NAME=$1

# 工作目录 (本项目挂载根目录)
pushd /workspace/code
# tpu_mlir sdk 环境
source env/setup_tpu_mlir.sh

# 编译模型保存目录
if [ ! -d models ]; then mkdir models; fi
cd models
if [ ! -d $MODEL_NAME ]; then
  echo ">> model subfolder $$MODEL_NAME not found, should run other init script first :("
  exit -1
fi
cd $MODEL_NAME

# TPU批处理无并行加速效果 :(
B=$2 || 1
# 通道数 (y_only?)
C=$3 || 3
# 确定输入图片尺寸
H=$4 || 192
W=$5 || 256
# 唯一后缀
SUFFIX=${B}x${C}x${H}x${W}
# 设备
DEVICE=bm1684x
# 文件路径
MODEL_PATH=${MODEL_NAME}_${SUFFIX}.pt
MLIR_NAME=${MODEL_NAME}_${SUFFIX}.mlir
BMODEL_FP16_FILE=${MODEL_NAME}_${SUFFIX}.bmodel

# 将 torch.jit 模型转换为 mlir
if [ ! -f $MLIR_NAME ]; then
  model_transform.py \
    --model_name $MODEL_NAME \
    --input_shape [[$B,$C,$H,$W]] \
    --model_def $MODEL_PATH \
    --mlir $MLIR_NAME
fi

# 将 mlir 转换成 fp16 的 bmodel
if [ ! -f $BMODEL_FP16_FILE ]; then
  model_deploy.py \
    --mlir $MLIR_NAME \
    --quantize F16 \
    --chip $DEVICE \
    --model $BMODEL_FP16_FILE
fi
