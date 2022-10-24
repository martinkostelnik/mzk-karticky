#!/bin/bash

BASE=/home/xkoste12/mzk-karticky

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/src/NER
DATA_DIR=/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-ocr
OUT_DIR=$BASE/inference-output
MODEL_DIR=$BASE/experiments/2022-10-14/e12/checkpoints

mkdir -p $OUT_DIR

python -u $SCRIPTS_DIR/inference.py \
    --model-path=$MODEL_DIR/checkpoint_008.pth \
    --config-path=$MODEL_DIR \
    --tokenizer-path=$MODEL_DIR \
    --data-path=$DATA_DIR \
    --save-path=$OUT_DIR
