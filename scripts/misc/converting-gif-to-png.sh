#!/usr/bin/env bash
#$ -l matylda5=2.5
#$ -S bash

# Author: Karel Beneš

SOURCE_DIR=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky
TARGET_DIR=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-png

readarray -t data_dirs </mnt/matylda5/ibenes/projects/pero/MZK-karticky/data_dirs_high_level.txt

arr_idx=$((SGE_TASK_ID-1))

line=${data_dirs[$arr_idx]}

cd $SOURCE_DIR/$line

for suplik in su*
do
    cd $SOURCE_DIR/$line/$suplik
    for f in *.gif 
    do
        png_dest=$TARGET_DIR/$line/$suplik
        mkdir -p $png_dest

        convert $f $png_dest/$f.png
    done
done
