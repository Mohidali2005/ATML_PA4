#!/bin/bash
# clones the course data repository at a fixed commit and builds the locked task files
# usage: bash scripts/get_data.sh
set -e
commit=6a0a9ec9da7ef3d313ee3036cfec493886723644
dest=data/course
if [ ! -d $dest/.git ]; then
  git clone -q https://github.com/AbDu11aHHH/atml-pa4-datasets.git $dest
fi
git -C $dest checkout -q $commit
# reusing our pinned source revisions so every run builds identical files
if [ -f configs/data_lock/local_source_revisions.json ]; then
  cp configs/data_lock/local_source_revisions.json $dest/data/local_source_revisions.json
fi
python $dest/scripts/setup_data.py --public
python $dest/scripts/check_setup.py
python -m task1_compress.data --verify
