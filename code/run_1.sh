#!/bin/bash

set -exu

python3 -u -m code.impl_1 --data-path data/wiki-sentences.txt --model-output-path gen/base_model ${OPTIONS} | tee code/out_1.txt

