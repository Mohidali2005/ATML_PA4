#!/bin/bash
# builds the data then runs a short smoke pass the two parallel tracks and the timing stage
# usage: bash scripts/run_task1_kaggle.sh <artifacts_dir>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
art=$1
logs=results/task1/logs
mkdir -p $logs
set -e
bash scripts/get_data.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# stopping here if any stage fails on the small sizes
CUDA_VISIBLE_DEVICES=0 python -m task1_compress.run_task1 --stage all --smoke --artifacts $art/smoke > $logs/smoke.log 2>&1
set +e
CUDA_VISIBLE_DEVICES=0 python -m task1_compress.run_task1 --stage track_a --artifacts $art/full > $logs/track_a.log 2>&1 &
a=$!
CUDA_VISIBLE_DEVICES=1 python -m task1_compress.run_task1 --stage track_b --artifacts $art/full > $logs/track_b.log 2>&1 &
b=$!
wait $a
ea=$?
wait $b
eb=$?
echo "track a exit $ea track b exit $eb"
if [ $ea -ne 0 ] || [ $eb -ne 0 ]; then
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 python -m task1_compress.run_task1 --stage final --artifacts $art/full > $logs/final.log 2>&1
echo "final exit $?"
echo "finished task1"
