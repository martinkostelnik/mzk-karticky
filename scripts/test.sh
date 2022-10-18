#!/bin/bash

BASE=/home/xkoste12/mzk-karticky

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/src/NER
DATA_DIR=$BASE/data
MZK_DIR=/mnt/xkoste12/matylda1/ikiss/data/mzk_karticky/2022-09-19
E_DIR=$BASE/experiments/2022-10-14/e12/checkpoints

python -u $SCRIPTS_DIR/test.py \
    --model-path=$E_DIR/checkpoint_008.pth \
    --config-path=$E_DIR \
    --tokenizer-path=$E_DIR \
    --ocr-path=$DATA_DIR/page-txts \
    --data-path=$BASE/test-inference-output/dataset.all
