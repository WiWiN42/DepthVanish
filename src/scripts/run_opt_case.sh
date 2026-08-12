#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

python src/opt_case.py \
--exp test/aanet_test \
--round 500 \
--n_checkpoint 10 \
--save_dir result \
--log_level info \
--n_xtile 5 \
--n_ytile 4 \
--physical_height 0.891 \
--physical_width 1.26 \
--physical_depth 5 \
--physical_shiftx 0 \
--physical_shifty 0 \
--unit_norm 0.5 \
--alpha 0.1 \
--gamma 10 \
--model aanet \
--ckpt /home/yxing/projects/stereo_PhysicalAttack/src/model/_checkpoints/aanet_kitti15-fb2a0d23.pth \
--img_norm_min 0 \
--img_norm_max 1 \
--dataset KITTI \
--img_left /mnt/data/data_yxing/KITTI_stereo_2015/training/image_2/000003_10.png \
--img_right /mnt/data/data_yxing/KITTI_stereo_2015/training/image_3/000003_10.png \
--stereo_calib /mnt/data/data_yxing/KITTI_stereo_2015/training/calib_cam_to_cam/000003.txt \