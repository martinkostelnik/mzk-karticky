#!/usr/bin/bash

# Author: Martin Kostelník
# Brief: Creates an index containing bib records used in card matching.

BASE=/home/xkoste12/mzk-karticky
BIB_PATH=/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/bibliotheke-records/mzk_all.full-id.txt

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

INDEX_BUILD_SCRIPT=$BASE/src/matching/index_build_bib.py
INDEX_DIR=$BASE/data/index-bib-normalized

mkdir -p $INDEX_DIR

python $INDEX_BUILD_SCRIPT \
    --index-dir $INDEX_DIR \
    --bib-file $BIB_PATH
