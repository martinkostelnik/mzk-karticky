#!/usr/bin/env bash
#$ -l ram_free=1G
#$ -l mem_free=1G
#$ -l matylda5=1.5
#$ -S bash

echo $(hostname)

readarray -t data_dirs </mnt/matylda5/ibenes/projects/pero/MZK-karticky/data_dirs.txt

arr_idx=$((SGE_TASK_ID-1))

line=${data_dirs[$arr_idx]}

ocr_xml_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-ocr/$line

for f in $ocr_xml_dir/*.xml
do
    cat $f | grep 'Unicode' | sed 's/^[[:space:]]*<Unicode>//' | rev | sed 's/^[[:space:]]*>edocinU\/<//' | rev  > $f.txt
done
