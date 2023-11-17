#!/usr/bin/env bash

# 工作目录 (本项目挂载根目录)
pushd /workspace/code
# tpu_mlir sdk 环境
source setup_tpu_mlir.sh

# 编译模型保存目录
if [ ! -d models ]; then mkdir models; fi
cd models
if [ ! -d ninasr_b0_x4 ]; then
  echo ">> please run: python convert_torchsr.py"
  exit -1
fi
cd ninasr_b0_x4

# 确定输入大小
B=4
H=192
W=256

# 文件路径
DATA_PATH=/workspace/code/data/test
MODEL_PATH=ninasr_b0_x4.pt
MLIR_NAME=ninasr.mlir
BMODEL_FP32_FILE=ninasr.fp32.bmodel
BMODEL_INT8_FILE=ninasr.int8.bmodel
BMODEL_CALIB_FILE=ninasr.int8.cali

# 将 torch.jit 模型转换为 mlir
if [ ! -f $MLIR_NAME ]; then
  model_transform.py \
    --model_name ninasr \
    --input_shape [[$B,3,$H,$W]] \
    --model_def $MODEL_PATH \
    --mlir $MLIR_NAME
fi

# 将 mlir 转换成 fp32 的 bmodel
if [ ! -f $BMODEL_FP32_FILE ]; then
  model_deploy.py \
    --mlir $MLIR_NAME \
    --quantize F32 \
    --chip bm1684x \
    --model $BMODEL_FP32_FILE
fi

# 制作校准表
if [ ! -f $BMODEL_CALIB_FILE ]; then
  run_calibration.py \
    $MLIR_NAME \
    --dataset $DATA_PATH \
    --input_num 100 \
    -o $BMODEL_CALIB_FILE
fi

# 将 mlir 转换成 int8 的 bmodel
if [ ! -f $BMODEL_INT8_FILE ]; then
  model_deploy.py \
    --mlir $MLIR_NAME  \
    --quantize INT8 \
    --calibration_table $BMODEL_CALIB_FILE  \
    --chip bm1684x \
    --tolerance 0.85,0.45 \
    --model $BMODEL_INT8_FILE
fi

# list up generated files
ls -l
