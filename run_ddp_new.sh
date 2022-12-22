#!/bin/bash

export OMP_NUM_THREADS=4 # per gpu
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# --train_dir ./Datasets/DarkFace_Train_2021/image \

SEED=42
N_GPU=8
BS=1
BS_TOTAL=$((BS * N_GPU))
EPOCH=50
LR=1e-5
SCHE_STEP=1
STAGE=3
TIME=$(date +%Y-%m-%d_%H:%M:%S)
INFO=baseline_test_ddp

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node ${N_GPU} train_ddp.py \
--global_rank 0 \
--eval \
--wandb \
--inception \
--batch_size ${BS} \
--seed ${SEED} \
--epochs ${EPOCH} \
--lr ${LR} \
--schedular_step ${SCHE_STEP} \
--stage ${STAGE} \
--train_dir ../image \
--test_dir ./data/all \
--output_dir ./results \
--model_save_per_epoch=10 \
--eval_per_epoch=1 \
--start_time=${TIME} \
--exp_name=${INFO}_lr_${LR}_sche_${SCHE_STEP}_stage_${STAGE}_epoch_${EPOCH}


# copy all the image in directory ./A to a combined folder ./B
find ./A -type f -exec cp {} ./B \;