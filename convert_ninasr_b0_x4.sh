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
B=1
H=192
W=256

# 将 torch.jit 模型转换为 mlir
if [ ! -f ninasr_b0_x4.mlir ]; then
  model_transform.py \
   --model_name r-esrgan \
   --input_shape [[$B,3,$H,$W]] \
   --model_def ninasr_b0_x4.pt \
   --mlir ninasr_b0_x4.mlir
fi

# 将 mlir 转换成 fp32 的 bmodel
if [ ! -f ninasr_b0_x4.fp32.bmodel ]; then
  model_deploy.py \
  --mlir ninasr_b0_x4.mlir \
  --quantize F32 \
  --chip bm1684x \
  --model ninasr_b0_x4.fp32.bmodel
fi

# 制作校准表
if [ ! -f ninasr_b0_x4.int8.cali ]; then
  run_calibration.py \
    ninasr_b0_x4.mlir \
    --dataset /workspace/code/data/test \
    --input_num 100 \
    -o ninasr_b0_x4.int8.cali
fi

# 将 mlir 转换成 int8 的 bmodel
if [ ! -f ninasr_b0_x4.int8.bmodel ]; then
  model_deploy.py \
    --mlir ninasr_b0_x4.mlir  \
    --quantize INT8 \
    --calibration_table ninasr_b0_x4.int8.cali  \
    --chip bm1684x \
    --tolerance 0.85,0.45 \
    --model ninasr_b0_x4.int8.bmodel
fi

# list up generated files
ls -l
