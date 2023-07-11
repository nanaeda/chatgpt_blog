#!/bin/bash

set -exu

python3 -u -m code.impl_3 --data-path data/wiki-sentences.txt --model-output-path gen/base_model --use-positional-embedding ${OPTIONS} | tee code/out_3.txt
