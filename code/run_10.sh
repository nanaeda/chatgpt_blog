#!/bin/bash

set -exu

python3 -u -m code.impl_10 \
  --prompt-data-path data/japanese_alpaca_data.json \
  --sft-model-path gen/sft_model \
  --reward-model-path gen/reward_model \
  --model-output-path gen/instruct_gpt_model \
  --bpe-path data/bpe.txt ${OPTIONS} | tee code/out_10.txt
