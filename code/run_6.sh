#!/bin/bash

set -exu

python3 -u -m code.impl_6 --data-path data/wiki-sentences.txt --model-output-path gen/base_model --use-positional-embedding --bpe-path data/bpe.txt ${OPTIONS} | tee code/out_6.txt
