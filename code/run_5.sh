#!/bin/bash

set -exu

python3 -u -m code.impl_5 --data-path data/wiki-sentences.txt --vocabruary-size 16384 --merge-rule-output-path data/bpe.txt | tee code/out_5.txt
