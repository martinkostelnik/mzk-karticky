#!/bin/bash

BASE=/home/xkoste12/mzk-karticky

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/src/NER
DATA_DIR=$BASE/data
OUT_DIR=$BASE/test-inference-output
E_DIR=$BASE/experiments/2022-10-14/e12/checkpoints

mkdir -p $OUT_DIR

python -u $SCRIPTS_DIR/inference.py \
    --model-path=$E_DIR/checkpoint_008.pth \
    --config-path=$E_DIR \
    --tokenizer-path=$E_DIR \
    --data-path=$DATA_DIR/page-txts \
    --save-path=$OUT_DIR \
    --aggfunc prod \
    --threshold 0.0
