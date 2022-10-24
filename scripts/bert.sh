#!/bin/bash

BASE=/home/xkoste12/mzk-karticky

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/src/bert
DATA_DIR=/mnt/xkoste12/matylda1/ikiss/pero/experiments/bert_training/data/diakorp
OUT_DIR=$BASE/test-bert-training-output

mkdir -p $OUT_DIR

python train.py \
    --train-path=$DATA_DIR/lines.trn \
    --test-path=$DATA_DIR/lines.tst \
    --bert-path="bert-base-multilingual-uncased" \
    --tokenizer-path="bert-base-multilingual-uncased" \
    --batch-size=10 \
    --learning-rate=1e-4 \
    --masking-prob=0.15 \
    --iterations=20000 \
    --view-step=500 \
    --save-dir=$OUT_DIR
