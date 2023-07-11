#!/bin/bash

set -exu

python3 -u -m code.impl_7 \
  --prompt-data-path data/japanese_alpaca_data.json \
  --base-gpt-path gen/base_model \
  --model-output-path gen/sft_model \
  --bpe-path data/bpe.txt ${OPTIONS} | tee code/out_7.txt
