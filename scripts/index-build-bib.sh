#!/usr/bin/bash

BASE=/home/xkoste12/mzk-karticky
BIB_PATH=/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/bibliotheke-records/mzk_all.full-id.txt

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

INDEX_BUILD_SCRIPT=$BASE/kareluv-binec/index-build-bib-new.py
INDEX_DIR=$BASE/data/index

mkdir -p $INDEX_DIR

python $INDEX_BUILD_SCRIPT \
    --index-dir $INDEX_DIR \
    --bib-file $BIB_PATH
