#!/bin/bash

set -exu

python3 -u -m code.impl_0 --data-path data/wiki-sentences.txt | tee code/out_0.txt

