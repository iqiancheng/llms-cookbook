#!/bin/bash

PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
echo "PROJECT_DIR: $PROJECT_DIR"
# exit 0

LOG_DIR=$PROJECT_DIR/logs
echo "LOG_DIR: $LOG_DIR"
mkdir -p $LOG_DIR
ps -eo pid=,cmd= | awk '$0 ~ /\/python([0-9.]*)?[[:space:]]+-u[[:space:]]+[^[:space:]]+(\b|[[:space:]])/ {print $1}' | xargs -r kill -9
sleep 1
# export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT
export OMP_NUM_THREADS=1

# update ucx path: https://github.com/openucx/ucc/issues/476
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# run on 8 x H100/H800 GPUs
torchrun --nproc-per-node 8 examples/llm_finetune/finetune.py \
    --config recipes/nemo_automodel/qwen/qwen3_moe_30b_te_packed.yaml \
    2>&1 | tee $LOG_DIR/qwen3_moe_30b_te_packed_$(date +%Y%m%d_%H%M%S).log