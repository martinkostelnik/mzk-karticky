#!/usr/bin/env bash
#$ -l ram_free=9G
#$ -l mem_free=9G
#$ -l gpu=1
#$ -l matylda5=1.5
#$ -l h=!(pcspeech-gpu|supergpu3|supergpu4|supergpu6|supergpu8|supergpu9|supergpu10|supergpu11|supergpu12|supergpu13|supergpu14|supergpu15|supergpu16|supergpu17|supergpu18)
#$ -S bash

echo $(hostname)

PARSE_FOLDER=/mnt/matylda5/ibenes/teh_codez/pero-ocr/user_scripts/parse_folder.py

source $HOME/.bashrc
conda activate ultimate

readarray -t data_dirs </mnt/matylda5/ibenes/projects/pero/MZK-karticky/data_dirs.txt

arr_idx=$((SGE_TASK_ID-1))

line=${data_dirs[$arr_idx]}

img_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-png/$line

crop_xml_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-crop/$line
mkdir -p $crop_xml_dir

python3 $PARSE_FOLDER \
    -c /mnt/matylda5/ibenes/projects/pero/MZK-karticky/config-layout.ini \
    -i $img_dir \
    --output-xml-path $crop_xml_dir \
    --set-gpu \
    2>&1 | tee $crop_xml_dir/parse_folder.log


ocr_xml_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-ocr/$line
mkdir -p $ocr_xml_dir
logits_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-logits/$line
mkdir -p $logits_dir

python3 $PARSE_FOLDER \
    -c /mnt/matylda5/ibenes/projects/pero/MZK-karticky/config-ocr.ini \
    -i $img_dir \
    -x $crop_xml_dir \
    --output-xml-path $ocr_xml_dir \
    --output-logit-path $logits_dir \
    --output-transcriptions-file-path transcriptions \
    --set-gpu \
    2>&1 | tee $ocr_xml_dir/parse_folder.log
