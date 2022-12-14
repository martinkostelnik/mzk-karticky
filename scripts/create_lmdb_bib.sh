#!/usr/bin/bash

# Author: Martin Kostelník
# Brief: Creates a LMDB containing bib records used in card matching.

BASE=/home/xkoste12/mzk-karticky
LMDB_SCRIPT=$BASE/src/matching/create_lmdb_bib.py
BIB_PATH=/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/bibliotheke-records/mzk_all.full-id.txt
OUT_DIR=$BASE/data/lmdb-bib-normalized

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

mkdir -p $OUT_DIR

python $LMDB_SCRIPT \
    --out $OUT_DIR \
    --bib $BIB_PATH
