#!/usr/bin/env bash
python train_i3d.py --train_list data/hmdb51/hmdb51_flow_train_split_1.txt \
--val_list data/hmdb51/hmdb51_flow_val_split_1.txt \
--dataset hmdb51 \
--mode flow --arch bilinear_i3d \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 10 20 25 30 35 40 --epochs 45 --batch-size 3 \
--snapshot_pref checkpoints/ \
--dropout 0.3
