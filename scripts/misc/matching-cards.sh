#!/usr/bin/env bash
#$ -l ram_free=12G
#$ -l mem_free=12G
#$ -l matylda5=0.2
#$ -S bash

# Author: Karel Beneš
# Brief: Old script to run matching of OCR cards with database IDs

echo $(hostname)

INDEX_SEARCH=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/index-search-bib.py

source $HOME/.bashrc
conda activate ultimate

readarray -t data_dirs </mnt/matylda5/ibenes/projects/pero/MZK-karticky/data_dirs.txt

arr_idx=$((SGE_TASK_ID-1))

line=${data_dirs[$arr_idx]}

ocr_xml_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-ocr/$line
index_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/indexdir-bib-filtered-full_id
bib_pickle=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/mkz_all.full_id.filtered.pickle

match_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-full_id-matches-vFiltered_universal-all
mkdir -p $match_dir

su_name=$(echo $line | sed 's@/@-@g')
python3 $INDEX_SEARCH \
    --logging-level INFO \
    --index-dir $index_dir \
    --card-dir $ocr_xml_dir \
    --bib-pickle $bib_pickle \
    --min-matched-lines 1 \
    > $match_dir/$su_name
