#!/bin/bash
#SBATCH -N 1
#SBATCH -p a10
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:4
#SBATCH --array=0
#SBATCH -J llie-sci
#SBATCH -o ./slurm_logs/llie-sci-%A-%a.log
#SBATCH -e ./slurm_logs/llie-sci-%A-%a.log

SLURM_ARRAY_TASK_ID=0

case $SLURM_ARRAY_TASK_ID in
0)
  ICP=0
  FL=1
  SL=1
  ;;
1)
  ICP=0
  FL=1
  SL=0
  ;;
2)
  ICP=0
  FL=0
  SL=1
  ;;
3)
  ICP=0
  FL=2
  SL=1
  ;;
4)
  ICP=0
  FL=1
  SL=2
  ;;
5)
  ICP=1
  FL=1
  SL=1
  ;;
6)
  ICP=1
  FL=1
  SL=0
  ;;
7)
  ICP=1
  FL=0
  SL=1
  ;;
8)
  ICP=1
  FL=2
  SL=1
  ;;
9)
  ICP=1
  FL=1
  SL=2
  ;;
*)
  echo "Error: Invalid SLURM_ARRAY_TASK_ID"
  exit 1
  ;;
esac

export OMP_NUM_THREADS=4 # per gpu
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# --train_dir ./Datasets/DarkFace_Train_2021/image \

SEED=42
N_GPU=4
BS=1
BS_TOTAL=$((BS * N_GPU))
EPOCH=2
LR=1e-5
SCHE_STEP=1
STAGE=3
TIME=$(date +%Y-%m-%d_%H:%M:%S)
INFO=ablation_test

torchrun --nproc_per_node ${N_GPU} train_ddp.py \
--global_rank 0 \
--eval \
--wandb \
--dryrun \
--inception ${ICP} \
--batch_size ${BS} \
--seed ${SEED} \
--epochs ${EPOCH} \
--lr ${LR} \
--schedular_step ${SCHE_STEP} \
--stage ${STAGE} \
--train_dir ./Datasets/combined \
--output_dir ./results \
--test_dir ./Datasets/combined_test/low \
--test_gt_dir ./Datasets/combined_test/normal \
--fidelityloss 1 \
--smoothloss 1 \
--model_save_per_epoch=10 \
--eval_per_epoch=1 \
--start_time=${TIME} \
--exp_name=${INFO}_bs_${BS_TOTAL}_lr_${LR}_sche_${SCHE_STEP}_stage_${STAGE}_epoch_${EPOCH}_fl_${FL}_sl_${SL}_icp_${ICP}


# find ./GladNet-Dataset/Normal -type f -exec cp {} ./combined_test/normal \;
# how to re