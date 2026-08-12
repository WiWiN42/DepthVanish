#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

python src/opt_dataset.py \
--exp opt_dataset/raftstereo_foggy_opt \
--round 300 \
--n_checkpoint 50 \
--save_dir result \
--log_level info \
--n_xtile 5 \
--n_ytile 4 \
--physical_height 0.891 \1
--physical_width 1.26 \
--physical_depth 5 \
--physical_shiftx 0 \
--physical_shifty 0 \
--unit_norm 0.5 \
--alpha 0.1 \
--gamma 10 \
--model raftstereo \
--ckpt /home/yxing/projects/stereo_PhysicalAttack/src/model/_checkpoints/raftstereo-sceneflow.pth \
--img_norm_min 0 \
--img_norm_max 1 \
--dataset DrivingStereo \
--img_left_file /home/yxing/projects/stereo_PhysicalAttack/data/drivingstereo_foggy_val_image0.txt \
--img_right_file /home/yxing/projects/stereo_PhysicalAttack/data/drivingstereo_foggy_val_image1.txt \
