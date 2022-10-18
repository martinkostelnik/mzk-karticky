#!/bin/bash

BASE=/home/xkoste12/mzk-karticky

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/src/NER
DATA_DIR=$BASE/data
MZK_DIR=/mnt/xkoste12/matylda1/ikiss/data/mzk_karticky/2022-09-19
OUT_DIR=$BASE/test-training-output

mkdir -p $OUT_DIR

python -u $SCRIPTS_DIR/train.py \
    --epochs=10 \
    --batch-size=10 \
    --train-bert \
    --ocr-path=$MZK_DIR/ocr.lmdb \
    --train-path=$DATA_DIR/alignment.test \
    --val-path=$DATA_DIR/alignment.test \
    --test-path=$DATA_DIR/alignment.test \
    --save-tokenizer \
    --save-path=$OUT_DIR \
    --min-aligned=4 \
    --must-align Author Title ID \
    --sep \
    --labels all \
    --format io
