#!/bin/bash
# clones the frozen course data then runs a smoke pass and the three parts of task 2 on two gpus
# usage: bash scripts/run_task2_kaggle.sh
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
# stopping here if any part fails on the small sizes
CUDA_VISIBLE_DEVICES=0 python -m task2_attention.exact --smoke --results outputs/task2/smoke > $logs/smoke_exact.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -m task2_attention.train_recall --smoke --results outputs/task2/smoke > $logs/smoke_recall.log 2>&1
CUDA_VISIBLE_DEVICES=1 python -m task2_attention.kv_eval --smoke --results outputs/task2/smoke > $logs/smoke_kv.log 2>&1
echo "smoke passed"
set +e
(
  CUDA_VISIBLE_DEVICES=0 python -m task2_attention.exact > $logs/exact.log 2>&1
  echo "exact exit $?"
  CUDA_VISIBLE_DEVICES=0 python -m task2_attention.train_recall > $logs/recall.log 2>&1
  echo "recall exit $?"
) &
a=$!
CUDA_VISIBLE_DEVICES=1 python -m task2_attention.kv_eval > $logs/kv.log 2>&1 &
b=$!
wait $a
wait $b
echo "kv exit $?"
python -m task2_attention.analyze > $logs/analyze.log 2>&1
echo "analyze exit $?"
echo "finished task2"
