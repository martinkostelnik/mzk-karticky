#!/usr/bin/bash

BASE=/home/xkoste12/mzk-karticky
BIB_PATH=/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/bibliotheke-records/mzk_all.full-id.txt
LMDB_PATH=/mnt/xkoste12/matylda1/ikiss/data/mzk_karticky/2022-09-19/ocr.lmdb

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

INDEX_SEARCH_SCRIPT=$BASE/kareluv-binec/index-search-bib-new.py
INDEX_DIR=$BASE/data/index
INFERENCE_PATH=$BASE/test-inference/inference.all
OUT_DIR=$BASE/matching_output

mkdir -p $OUT_DIR

python $INDEX_SEARCH_SCRIPT \
    --index-dir $INDEX_DIR \
    --lmdb-path $LMDB_PATH \
    --bib-file $BIB_PATH \
    --inference-path $INFERENCE_PATH \
    --out-path $OUT_DIR \
    --min-matched-lines 3
