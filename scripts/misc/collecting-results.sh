#!/usr/bin/env bash

# Author: Karel Beneš

match_dir=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky-full_id-matches-vFiltered_universal-all/

cd $match_dir

for f in *
do
    sed "s/^/$f-/" <$f >$f.full_id
done

cat *.full_id > all.txt
cat all.txt | sort -nr -k 3 > all.sorted.txt
