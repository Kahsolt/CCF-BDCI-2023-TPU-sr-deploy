#!/usr/bin/env bash

# 模型名
MODEL_NAME=espcn_ex

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

# TPU并不支持并行处理批
B=1
# 确定输入图片尺寸
C=3
H=192
W=256
# 设备
DEVICE=bm1684x
# 文件路径
DATA_PATH=/workspace/code/data/test
MY_CALIB_SCRIPT=/workspace/code/run_calibration_y_only.py
MODEL_PATH=$MODEL_NAME.pt
MLIR_NAME=$MODEL_NAME.mlir
CALIB_FILE=$MODEL_NAME.cali
BMODEL_INT8_FILE=$MODEL_NAME.int8.bmodel

# 将 torch.jit 模型转换为 mlir
if [ ! -f $MLIR_NAME ]; then
  model_transform.py \
    --model_name $MODEL_NAME \
    --input_shape [[$B,$C,$H,$W]] \
    --model_def $MODEL_PATH \
    --mlir $MLIR_NAME
fi

# 制作校准表
if [ ! -f $CALIB_FILE ]; then
  run_calibration.py \
    $MLIR_NAME \
    --dataset $DATA_PATH \
    --input_num 100 \
    -o $CALIB_FILE
fi

# 将 mlir 转换成 int8 的 bmodel
if [ ! -f $BMODEL_INT8_FILE ]; then
  model_deploy.py \
    --mlir $MLIR_NAME \
    --quantize INT8 \
    --calibration_table $CALIB_FILE  \
    --tolerance 0.99,0.90 \
    --chip $DEVICE \
    --model $BMODEL_INT8_FILE
fi

# list up generated files
ls -l
