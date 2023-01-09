#!/usr/bin/bash

# Author: Martin Kostelník
# Brief: Compares two matching files.

BASE=/home/xkoste12/mzk-karticky

SCRIPT=$BASE/src/matching/compare_matching.py
NEW_PATH=$BASE/matching_output_tmp/matching.txt
OLD_PATH=/mnt/xkoste12/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-matches-v1/all.txt

source $BASE/venv/bin/activate
export PATH="$BASE/venv/bin:$PATH"

python $SCRIPT \
    --new $NEW_PATH \
    --old $OLD_PATH \
