#!/bin/bash
python train.py --batch_size 8 \
--gpu 1 \
--seed 42 \
--epochs 100 \
--lr 3e-4 \
--stage 3 \
--train_dir ./data/medium \
--test_dir ./data/medium \
--save ./results \
--model ./weights/medium.pt


