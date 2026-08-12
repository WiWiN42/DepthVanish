#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

python src/grid_opt.py \
--exp grid_opt/raftstereo \
--round 300 \
--n_checkpoint 20 \
--save_dir result \
--log_level info \
--physical_height 0.891 \
--physical_width 1.26 \
--physical_depth 4.9 \
--physical_shiftx 0 \
--physical_shifty 0 \
--unit_norm 0.5 \
--model raftstereo \
--ckpt /home/yxing/projects/stereo_PhysicalAttack/src/model/_checkpoints/raftstereo-sceneflow.pth \
--img_norm_min 0 \
--img_norm_max 1 \
--dataset KITTI \
--img_left /mnt/data/data_yxing/KITTI_stereo_2015/training/image_2/000003_10.png \
--img_right /mnt/data/data_yxing/KITTI_stereo_2015/training/image_3/000003_10.png \
--stereo_calib /mnt/data/data_yxing/KITTI_stereo_2015/training/calib_cam_to_cam/000003.txt \