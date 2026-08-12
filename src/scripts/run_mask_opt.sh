#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

python src/mask_opt.py \
--exp test/raftstereo_test \
--round 500 \
--n_checkpoint 10 \
--save_dir result \
--log_level info \
--n_xtile 5 \
--n_ytile 4 \
--relax_temp 0.5 \
--grad_norm_start 1 \
--grad_norm_end 0.01 \
--grad_step_interval 100 \
--bin_norm_start 0.5 \
--bin_norm_end 1.5 \
--binary_step_interval 100 \
--alpha 0.1 \
--gamma 10 \
--model raftstereo \
--ckpt /home/yxing/projects/stereo_PhysicalAttack/src/model/_checkpoints/raftstereo-sceneflow.pth \
--img_norm_min 0 \
--img_norm_max 1 \
--img_left /mnt/data/data_yxing/KITTI_stereo_2015/training/image_2/000003_10.png \
--img_right /mnt/data/data_yxing/KITTI_stereo_2015/training/image_3/000003_10.png \