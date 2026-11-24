#!/bin/bash
# trains the controlled recall intervention and logs key and value statistics for the cache study on two gpus
# usage: bash scripts/run_task2_b3_kaggle.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
logs=results/task2/logs
mkdir -p $logs
set -e
commit=6a0a9ec9da7ef3d313ee3036cfec493886723644
dest=data/course
if [ ! -d $dest/.git ]; then
  git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git $dest
fi
git -C $dest checkout -q $commit
python -m task2_attention.mqar --verify
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# stopping here if the new model fails on the small sizes
CUDA_VISIBLE_DEVICES=0 python -m task2_attention.train_recall --smoke --runs local_linear_mixed_k16 --stage train --results outputs/task2/smoke > $logs/smoke_b3.log 2>&1
echo "smoke passed"
set +e
(
  CUDA_VISIBLE_DEVICES=0 python -m task2_attention.train_recall --runs local_linear_mixed_k16 --stage train > $logs/b3.log 2>&1
  echo "b3 exit $?"
  # timing every mechanism again in one stage so the sweep stays a single consistent measurement
  CUDA_VISIBLE_DEVICES=0 python -m task2_attention.train_recall --stage sweep > $logs/sweep.log 2>&1
  echo "sweep exit $?"
) &
a=$!
CUDA_VISIBLE_DEVICES=1 python -m task2_attention.kv_eval --diagnostics > $logs/kv_diagnostics.log 2>&1
echo "diagnostics exit $?"
wait $a
echo "finished task2 b3"
