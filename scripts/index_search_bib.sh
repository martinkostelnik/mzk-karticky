#!/usr/bin/bash

# Author: Martin Kostelník
# Brief: Run matching of cards in sequential mode. This creates two files:
#   1. "matching.txt" containing matching of OCR file with DB ID
#   2. "alignment.txt" containing alignments for each OCR card

BASE=/home/xkoste12/mzk-karticky
OCR_LMDB_PATH=$BASE/data/lmdb
BIB_LMDB_PATH=$BASE/data/lmdb-bib-normalized
INDEX_SEARCH_SCRIPT=$BASE/src/matching/index_search_bib.py
INDEX_DIR=$BASE/data/index-bib-normalized-joint-long
INFERENCE_PATH=$BASE/inference-2022-10-27-e12/dataset.all
OUT_DIR=$BASE/matching_output_tmp

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

mkdir -p $OUT_DIR

python $INDEX_SEARCH_SCRIPT \
    --index-dir $INDEX_DIR \
    --ocr-lmdb $OCR_LMDB_PATH \
    --bib-lmdb $BIB_LMDB_PATH \
    --inference-path $INFERENCE_PATH \
    --out-path $OUT_DIR \
    --min-matched-lines 4
