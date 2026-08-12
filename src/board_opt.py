# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/03/31 10:10:32
@Author  :   yxing
"""

import os
import sys
import logging
import argparse
from argparse import Namespace
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

sys.path.append("/home/yxing/projects/stereo_PhysicalAttack/tools/RAFT-Stereo")
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

from utils.util import load_image, get_board_spec, round_numbers, save_tensor_to_img
from utils.assemble import DPABoard, assemble_board_into_stereo
from utils.loss import regional_mean_square_error

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--log_level", type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help="the log level for python logger")
parser.add_argument("--exp", type=str, default="test", help="the exp name")
parser.add_argument("--n_round", type=int, default=10, help="the rounds for searching an effective patch")
parser.add_argument("--save_dir", type=str, default='./result/', help="the directory to save middle results")

parser.add_argument("--board_size", type=tuple, default=(180,180))

parser.add_argument("--base_strip", type=str, default="/home/yxing/projects/stereo_PhysicalAttack/assets/textures/texture_seed_strip.png", help="the base strip to create the board")
parser.add_argument("--img_left", type=str, default="/mnt/data/data_yxing/KITTI_stereo_2015/training/image_2/000003_10.png", help="the left image of stereo pair")
parser.add_argument("--img_right", type=str, default="/mnt/data/data_yxing/KITTI_stereo_2015/training/image_3/000003_10.png", help="the right image of stereo pair")

parser.add_argument("--ckpt", type=str, default="/home/yxing/projects/stereo_PhysicalAttack/tools/RAFT-Stereo/models/raftstereo-sceneflow.pth", help="the pre-trained depth estimation model")

args = parser.parse_args()

# setup device
device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

# path work
# setup save root first
save_dir = os.path.join(args.save_dir, args.exp)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# strip_dir = os.path.join(save_dir, 'strip')
# Path(strip_dir).mkdir(parents=True, exist_ok=False)
board_dir = os.path.join(save_dir, 'board')
Path(board_dir).mkdir(parents=True, exist_ok=True)
depth_dir = os.path.join(save_dir, 'depth')
Path(depth_dir).mkdir(parents=True, exist_ok=True)

# setup logger
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("DPA")
logger.setLevel(getattr(logging, args.log_level.upper()))
fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
# fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
# fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# create & load the depth estimation model
model_para = Namespace(
    restore_ckpt=args.ckpt, 
    save_numpy=False, 
    # left_imgs='/home/yxing/projects/stereo_PhysicalAttack/cropr1_pasted.png', 
    # right_imgs='/mnt/data/data_yxing/KITTI/training/image_3/000005_10.png', 
    output_directory='demo_output',
    mixed_precision=True,
    valid_iters=10,
    hidden_dims=[128, 128, 128],
    corr_implementation='alt',
    shared_backbone=False,
    corr_levels=4,
    corr_radius=4,
    n_downsample=2,
    context_norm='batch',
    slow_fast_gru=False,
    n_gru_layers=3
)
model = torch.nn.DataParallel(RAFTStereo(model_para), device_ids=[0])
model.load_state_dict(torch.load(model_para.restore_ckpt, weights_only=True))
model = model.module
model.to(device)
model.eval()
logger.info("Model Loaded Successfully!")

# load stereo images into tensor
imgL, imgR = load_image(args.img_left), load_image(args.img_right)
logger.info("Stereo Images Loaded Successfully!")

# get the position and size of the board in the scene
pos_left, pos_right, boardL_size, boardR_size = get_board_spec(phy_size=(0.3,0.4), phy_depth=5, center_shift=(0,0)) #-TODO rewrite to strictly align with physical specification
strip_size = (boardL_size[0], boardL_size[1]/18) # empirical (h,w)
# round the specification
pos_left, pos_right, board_size, strip_size = round_numbers(*pos_left), round_numbers(*pos_right), round_numbers(*boardL_size), round_numbers(*strip_size)
logger.debug("The rounded board specification:")
logger.debug(f"pos_left --> {pos_left}")
logger.debug(f"pos_right --> {pos_right}")
logger.debug(f"board_size --> {board_size}")
logger.debug(f"strip_size --> {strip_size}")

# create the board base and optimizable strip
board_factory = DPABoard(board_size, strip_size, base_strip=args.base_strip, device=device)
opt_strip = torch.rand(size=(3,)+strip_size, dtype=torch.float32, device=device)
logger.info("Optimizable Strip Created Successfully!")

logger.info("step [1]: assemble strip into the board base")
# assemble the optimizable strip into the board
opt_board = board_factory.get_opt_board(opt_strip)
opt_board.requires_grad_(True)

# create an optimizer
strip_optimizer = torch.optim.Adam([opt_strip], lr=0.1)

for i in range(args.n_round):
    logger.info("\nStrip Optimization: round {:03d}".format(i))

    logger.info("step [2]: assemble board into the stereo images")
    # assemble the optimizable stereo images
    imgL, imgR = assemble_board_into_stereo(opt_board, imgL, imgR, pos_left, pos_right, board_size)
    imgL, imgR = imgL.to(device), imgR.to(device)

    logger.info("step [3]: pad image to fit input requirement")
    padder = InputPadder(imgL.shape, divis_by=32)
    image1_pad, image2_pad = padder.pad(imgL, imgR)

    logger.info("step [4]: estimate the disparity of the stereo images")
    # get estimation result
    _, flow_up = model(image1_pad, image2_pad, iters=model_para.valid_iters, test_mode=True)
    flow_up = padder.unpad(flow_up).squeeze()
    pred_disp = -flow_up.squeeze()

    logger.info("step [5]: calculate the regional disparity loss for the board")
    # calculate loss: mse, smooth, nps
    loss = regional_mean_square_error(pred_disp, region=pos_left) # the depth is anchor on left image

    logger.info("step [6]: backward the loss and update the optimizable strip")
    # optimize the optimizable strip
    strip_optimizer.zero_grad()
    loss.backward(retain_graph=True)
    strip_optimizer.step()

    logger.info("Optimization Done: loss {:05f}".format(loss))

    # save board and depth
    board = opt_board.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(board.transpose((1,2,0))).save(os.path.join(board_dir, 'round_{:03d}.jpg'.format(i)))

    depth = flow_up.detach().cpu().numpy().squeeze()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    Image.fromarray(depth).save(os.path.join(depth_dir, 'round_{:03d}.jpg'.format(i)))

    logger.info(f"Resulted board and depth are saved to {save_dir}\n")