#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python3 main.py \
--data_dir ../data/GOV/ \
--ckpt_dir ../GOV-m/ \
--summary_dir ../GOV-m/ \
--embedding_dim 100 \
--model_name GOV-m \
--margin_value 1 \
--batch_size 25000 \
--learning_rate 0.003 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 100 \
--max_epoch 5000
