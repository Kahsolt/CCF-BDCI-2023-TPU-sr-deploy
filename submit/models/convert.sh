#!/usr/bin/env bash

MODEL_NAME=$1
MODEL_PATH=${MODEL_NAME}.pt
MLIR_NAME=${MODEL_NAME}.mlir
BMODEL_FP16_FILE=${MODEL_NAME}.bmodel

model_transform.py \
  --model_name $MODEL_NAME \
  --input_shape [[1,3,128,128]] \
  --model_def $MODEL_PATH \
  --mlir $MLIR_NAME

model_deploy.py \
  --mlir $MLIR_NAME \
  --quantize F16 \
  --chip bm1684x \
  --model $BMODEL_FP16_FILE


rm *.npz *.mlir *.bmodel.json *.txt *.profile
