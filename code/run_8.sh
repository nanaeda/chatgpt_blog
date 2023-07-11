#!/bin/bash

set -exu

python3 -u -m code.impl_8 \
  --prompt-data-path data/japanese_alpaca_data.json \
  --sft-model-path gen/sft_model \
  --sample-output-path gen/sampled_answers.txt \
  --bpe-path data/bpe.txt ${OPTIONS} | tee code/out_8.txt
