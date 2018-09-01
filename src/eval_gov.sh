#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python3 eval.py \
--data_dir ../data/GOV/ \
--embedding_dim 120 \
--margin_value 1 \
--batch_size 20000 \
--learning_rate 0.005 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 100 \
--max_epoch 5000
