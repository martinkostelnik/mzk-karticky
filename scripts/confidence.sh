#!/bin/bash

BASE=/home/xkoste12/mzk-karticky
SCRIPTS_DIR=$BASE/src/NER
DATA_DIR=$BASE/data
OUT_DIR=$BASE/test-inference-output
MODEL_DIR=$BASE/experiments/2022-10-14/e12/checkpoints

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

mkdir -p $OUT_DIR
for function in prod mean min max median
do
    for threshold in 0.2 0.4 0.6 0.8 0.85 0.9 0.95 0.99
    do
        python -u $SCRIPTS_DIR/inference.py \
            --model-path=$MODEL_DIR/checkpoint_008.pth \
            --config-path=$MODEL_DIR \
            --tokenizer-path=$MODEL_DIR \
            --data-path=$DATA_DIR/page-txts \
            --save-path=$OUT_DIR \
            --aggfunc $function \
            --threshold $threshold

        python -u $SCRIPTS_DIR/compare_datasets.py
    done
done
