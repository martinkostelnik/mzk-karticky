#!/usr/bin/bash

BASE=/home/xkoste12/mzk-karticky
OCR_LMDB_PATH=$BASE/data/lmdb
BIB_LMDB_PATH=$BASE/data/lmdb-bib

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

INDEX_SEARCH_SCRIPT=$BASE/src/matching/index-search-bib-parallel.py
INDEX_DIR=$BASE/data/index
INFERENCE_PATH=$BASE/inference-2022-10-27-e12/dataset.all
OUT_DIR=$BASE/matching_output

mkdir -p $OUT_DIR

python $INDEX_SEARCH_SCRIPT \
    --index-dir $INDEX_DIR \
    --ocr-lmdb $OCR_LMDB_PATH \
    --bib-lmdb $BIB_LMDB_PATH \
    --inference-path $INFERENCE_PATH \
    --out-path $OUT_DIR \
    --min-matched-lines 4
