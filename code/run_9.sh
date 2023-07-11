#!/bin/bash

set -exu

python3 -u -m code.impl_9 \
  --sft-model-path gen/sft_model \
  --ranked-sample-path gen/ranked_answers.txt \
  --model-output-path gen/reward_model \
  --bpe-path data/bpe.txt ${OPTIONS} | tee code/out_9.txt
