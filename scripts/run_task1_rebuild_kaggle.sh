#!/bin/bash
# rebuilds every evaluated condition from the committed allocations then reruns the timing stage
# usage: bash scripts/run_task1_rebuild_kaggle.sh <artifacts_dir>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
art=$1
logs=results/task1/logs
mkdir -p $logs
set -e
bash scripts/get_data.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
CUDA_VISIBLE_DEVICES=0 python -m task1_compress.run_task1 --stage rebuild --artifacts $art/full > $logs/rebuild.log 2>&1
echo "rebuild exit $?"
CUDA_VISIBLE_DEVICES=0 python -m task1_compress.run_task1 --stage final --artifacts $art/full > $logs/final.log 2>&1
echo "final exit $?"
echo "finished task1 rebuild"
