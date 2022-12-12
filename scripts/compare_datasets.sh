#!/usr/bin/bash

# Author: Martin Kostelník
# Brief: Compares two datasets.

BASE=/home/xkoste12/mzk-karticky

SCRIPT=$BASE/src/NER/compare_datasets.py
LMDB_PATH=$BASE/data/lmdb
INFERENCE_PATH=$BASE/inference-2022-10-27-e12/dataset.alltrn
TRUTH_PATH=$BASE/data/alignment.good
MODEL_PATH=$BASE/experiments/2022-10-14/e12/checkpoints

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

python $SCRIPT \
    --inference $INFERENCE_PATH \
    --truth $TRUTH_PATH \
    --ocr $LMDB_PATH \
    --model $MODEL_PATH
