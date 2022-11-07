#!/usr/bin/env bash

WORKDIR=/mnt/matylda5/ibenes/projects/pero/MZK-karticky/all-karticky

mkdir -p $WORKDIR
cd $WORKDIR

tar -zxf '/mnt/matylda1/hradis/PERO/data/MZK - katalogove listky export/gkjpkz1.tar.gz' || exit 1
tar -zxf '/mnt/matylda1/hradis/PERO/data/MZK - katalogove listky export/gkjtkz2.tar.gz' || exit 1
tar -zxf '/mnt/matylda1/hradis/PERO/data/MZK - katalogove listky export/gkjukz1.tar.gz' || exit 1
tar -zxf '/mnt/matylda1/hradis/PERO/data/MZK - katalogove listky export/gkjukz2.tar.gz' || exit 1
