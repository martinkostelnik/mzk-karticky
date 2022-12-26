#!/bin/bash

# Author: Martin Kostelník
# Brief: Train NER model

BASE=/home/xkoste12/mzk-karticky

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

SCRIPTS_DIR=$BASE/src/NER
DATA_DIR=$BASE/data
MZK_DIR=$DATA_DIR/lmdb
OUT_DIR=$BASE/test-training-output
XML_DIR=$DATA_DIR/lmdb-pagexml

mkdir -p $OUT_DIR

python -u $SCRIPTS_DIR/train.py \
    --epochs=10 \
    --batch-size=10 \
    --train-bert \
    --ocr-path=$MZK_DIR \
    --train-path=$DATA_DIR/alignment.test \
    --val-path=$DATA_DIR/alignment.test \
    --test-path=$DATA_DIR/alignment.test \
    --save-tokenizer \
    --save-path=$OUT_DIR \
    --min-aligned=4 \
    --must-align Author Title ID \
    --sep \
    --labels subset \
    --format iob 
    # --xml-path=$XML_DIR \
    # --bboxes
    # --backend lambert
