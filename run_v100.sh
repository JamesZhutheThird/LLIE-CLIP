#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python train.py \
--eval \
--wandb \
--batch_size 4 \
--seed 42 \
--epochs 10 \
--lr 3e-4 \
--stage 3 \
--train_dir ../Datasets/DarkFace_Train_2021/image \
--test_dir ./data/all \
--output_dir ./results \
--exp_name baseline_test
