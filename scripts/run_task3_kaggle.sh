#!/bin/bash
# builds the data then runs a smoke pass and the four phases of task 3 on two gpus
# usage: bash scripts/run_task3_kaggle.sh <artifacts_dir>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
art=$1
logs=results/task3/logs
mkdir -p $logs
set -e
bash scripts/get_data.sh
python -m task3_speculative.data --verify --tokenizers
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"
# stopping here if any stage fails on the small sizes
small="--results outputs/task3/smoke --artifacts $art/smoke"
export CUDA_VISIBLE_DEVICES=0
(
  set -e
  python -m task3_speculative.checks --part real --smoke $small
  python -m task3_speculative.rollouts --stage pool --shard 0 --smoke $small
  python -m task3_speculative.rollouts --stage pool --shard 1 --smoke $small
  python -m task3_speculative.rollouts --stage shared --smoke $small
  python -m task3_speculative.distill --seed 6304 --smoke $small
  python -m task3_speculative.distill --seed 6305 --smoke $small
  python -m task3_speculative.merge $small
  python -m task3_speculative.rollouts --stage own --smoke $small
  python -m task3_speculative.alignment --stage prefixes --smoke $small
  python -m task3_speculative.quality --smoke $small
  python -m task3_speculative.benchmark --stage isolated --smoke $small
  python -m task3_speculative.benchmark --stage main --smoke $small
  python -m task3_speculative.alignment --stage benchmark --smoke $small
) > $logs/smoke.log 2>&1
echo "smoke passed"
rm -rf $art/smoke
set +e
full="--artifacts $art/full"
# phase 1 checks teacher responses and shared prefix banks
(
  export CUDA_VISIBLE_DEVICES=0
  python -m task3_speculative.checks --part real > $logs/checks.log 2>&1
  echo "checks exit $?"
  python -m task3_speculative.rollouts --stage pool --shard 0 > $logs/pool0.log 2>&1
  echo "pool 0 exit $?"
  python -m task3_speculative.rollouts --stage shared > $logs/shared.log 2>&1
  echo "shared banks exit $?"
) &
a=$!
CUDA_VISIBLE_DEVICES=1 python -m task3_speculative.rollouts --stage pool --shard 1 > $logs/pool1.log 2>&1 &
b=$!
wait $b
echo "pool 1 exit $?"
wait $a
# phase 2 both methods of one seed on each gpu
CUDA_VISIBLE_DEVICES=0 python -m task3_speculative.distill --seed 6304 $full > $logs/distill_6304.log 2>&1 &
a=$!
CUDA_VISIBLE_DEVICES=1 python -m task3_speculative.distill --seed 6305 $full > $logs/distill_6305.log 2>&1 &
b=$!
wait $a
ea=$?
wait $b
eb=$?
echo "distill 6304 exit $ea distill 6305 exit $eb"
if [ $ea -ne 0 ] || [ $eb -ne 0 ]; then
  exit 1
fi
# phase 3 merged checkpoints own prefix banks alignment and answer quality
CUDA_VISIBLE_DEVICES=0 python -m task3_speculative.merge $full > $logs/merge.log 2>&1
em=$?
echo "merge exit $em"
if [ $em -ne 0 ]; then
  exit 1
fi
(
  export CUDA_VISIBLE_DEVICES=0
  python -m task3_speculative.rollouts --stage own $full > $logs/own.log 2>&1
  echo "own banks exit $?"
  python -m task3_speculative.alignment --stage prefixes $full > $logs/alignment.log 2>&1
  echo "alignment exit $?"
) &
a=$!
CUDA_VISIBLE_DEVICES=1 python -m task3_speculative.quality $full > $logs/quality.log 2>&1 &
b=$!
wait $b
echo "quality exit $?"
wait $a
# phase 4 timing alone so that no other process disturbs it
export CUDA_VISIBLE_DEVICES=0
python -m task3_speculative.benchmark --stage isolated $full > $logs/isolated.log 2>&1
echo "isolated exit $?"
python -m task3_speculative.benchmark --stage main $full > $logs/benchmark.log 2>&1
echo "benchmark exit $?"
python -m task3_speculative.alignment --stage benchmark $full > $logs/surprisal.log 2>&1
echo "surprisal exit $?"
echo "finished task3"
