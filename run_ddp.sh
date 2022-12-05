#!/bin/bash

export OMP_NUM_THREADS=4 # per gpu
# export CUDA_VISIBLE_DEVICES=0,1,2,3

N_GPU=6
EPOCH=100
LR=1e-4
SCHE_STEP=1
STAGE=3
TIME=$(date +%Y-%m-%d_%H:%M:%S)
INFO=baseline_test_ddp

torchrun --nproc_per_node ${N_GPU} train_ddp.py \
--global_rank 0 \
--eval \
--wandb \
--batch_size 1 \
--seed 42 \
--epochs ${EPOCH} \
--lr ${LR} \
--schedular_step ${SCHE_STEP} \
--stage ${STAGE} \
--train_dir ./Datasets/DarkFace_Train_2021/image \
--test_dir ./data/all \
--output_dir ./results \
--model_save_per_epoch=10 \
--eval_per_epoch=1 \
--start_time=${TIME} \
--exp_name=${INFO}_lr_${LR}_sche_${SCHE_STEP}_stage_${STAGE}_epoch_${EPOCH}
