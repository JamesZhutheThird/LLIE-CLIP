#!/bin/bash

export OMP_NUM_THREADS=4 # per gpu
# export CUDA_VISIBLE_DEVICES=0,1,2,3

SEED=42
N_GPU=6
BS=1
BS_TOTAL=$((BS * N_GPU))
EPOCH=10
LR=1e-4
SCHE_STEP=1
STAGE=3
INFO=baseline_test_ddp
TRAIN_DIR=./data/all    #./Datasets/DarkFace_Train_2021/image
TEST_DIR=./data/all
OUTPUT_DIR=./results
TIME=$(date +%Y-%m-%d_%H:%M:%S)
EXP_NAME=${INFO}_bs_${BS_TOTAL}_lr_${LR}_sche_${SCHE_STEP}_stage_${STAGE}_epoch_${EPOCH}
THE_DIR=${OUTPUT_DIR}/${EXP_NAME}/${TIME}

torchrun --nproc_per_node ${N_GPU} train_ddp.py \
--global_rank 0 \
--eval \
--wandb \
--offline \
--batch_size ${BS} \
--seed ${SEED} \
--epochs ${EPOCH} \
--lr ${LR} \
--schedular_step ${SCHE_STEP} \
--stage ${STAGE} \
--train_dir ${TRAIN_DIR} \
--test_dir ${TEST_DIR} \
--output_dir ${OUTPUT_DIR} \
--model_save_per_epoch=10 \
--eval_per_epoch=1 \
--start_time=${TIME} \
--exp_name=${EXP_NAME}

find ${THE_DIR} -name offline-run* -type d | awk '{print wandb sync  }' | sh