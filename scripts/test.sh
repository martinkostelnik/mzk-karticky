#!/bin/bash

# Author: Martin Kostelník
# Brief: Test a NER model on the test dataset

BASE=/home/xkoste12/mzk-karticky
SCRIPTS_DIR=$BASE/src/NER
DATA_DIR=$BASE/data
MZK_DIR=$DATA_DIR/lmdb
XML_DIR=$DATA_DIR/lmdb-pagexml
E_DIR=$BASE/test-training-output

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"


python -u $SCRIPTS_DIR/test.py \
    --model-path=$E_DIR/checkpoint_010.pth \
    --config-path=$E_DIR \
    --tokenizer-path=$E_DIR \
    --ocr-path=$MZK_DIR \
    --data-path=$DATA_DIR/alignment.test \
    --xml-path=$XML_DIR
