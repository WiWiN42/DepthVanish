# -*- encoding: utf-8 -*-
"""
@File    :   mask_opt.py
@Time    :   2025/04/11 15:02:28
@Author  :   yxing
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import cv2

from utils.tool import save_img, save_depth, save_perturb, load_image, lp_projection
from utils.scheduler import Scheduler
from stereo_PhysicalAttack.src.utils.deploy import get_board_spec, round_numbers
from utils.assemble import DPABoard, embed_board_into_stereo, aggregate_board_grad
from utils.loss import regional_mean_square_error, regional_smooth_l1_loss, entropy_loss, tv_loss
from utils.dataset import StereoDataset
from model.stereo_model import StereoModel

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="test", help="the exp name")
parser.add_argument("--round", type=int, default=10, help="the number of rounds for searching an effective patch")
parser.add_argument("--n_checkpoint", type=int, default=20, help="the save interval")
parser.add_argument("--save_dir", type=str, default='./result/', help="the directory to save middle results")
parser.add_argument("--log_level", type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help="the log level for python logger")

parser.add_argument("--colored", action='store_true', help="whether set the mask as 3 channel colored or not")
parser.add_argument("--n_xtile", type=int, default=5, help="the number of horizontal tile to get the board")
parser.add_argument("--n_ytile", type=int, default=5, help="the number of horizontal tile to get the board")
parser.add_argument("--relax_temp", type=float, default=0.1, help="the temperature parameter to relax the mask")
parser.add_argument("--grad_norm_start", type=float, default=10, help="the starting value for constraint the mask gradients")
parser.add_argument("--grad_norm_end", type=float, default=0.01, help="the ending value for constraint the mask gradients")
parser.add_argument("--grad_step_interval", type=int, help="the number of iterations after which the gradient constraint is changed")
parser.add_argument("--bin_norm_start", type=float, default=10, help="the starting value for constraint the mask annealing")
parser.add_argument("--bin_norm_end", type=float, default=0.01, help="the ending value for constraint the mask annealing")
parser.add_argument("--binary_step_interval", type=int, help="the number of iterations after which the annealing constraint is changed")

parser.add_argument("--model", type=str, default='raft', help="the model to adopt")
parser.add_argument("--ckpt", type=str, help="the pre-trained depth estimation model")

parser.add_argument("--alpha", type=float, default=1, help="the weighting parameter of regional MSE of the board")
parser.add_argument("--beta", type=float, default=1, help="the weighting parameter of enhancing mask sharpness")
parser.add_argument("--gamma", type=float, default=1, help="the weighting parameter of encouraging smooth areas")
parser.add_argument("--delta", type=float, default=1, help="the weighting parameter of encouraging smooth areas")

# image normalization range
parser.add_argument("--img_norm_min", type=int, default=0, help="the minimum value to normalize the stereo images")
parser.add_argument("--img_norm_max", type=int, default=1, help="the maximum value to normalize the stereo images")

# stereo pair data
parser.add_argument("--img_left", type=str, default="/mnt/data/data_yxing/KITTI_stereo_2015/training/image_2/000003_10.png", help="the left image of stereo pair to demo")
parser.add_argument("--img_right", type=str, default="/mnt/data/data_yxing/KITTI_stereo_2015/training/image_3/000003_10.png", help="the right image of stereo pair to demo")

args = parser.parse_args()

# setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# setup save root first
save_dir = os.path.join(os.getcwd(), args.save_dir, args.exp)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
board_dir = os.path.join(save_dir, 'board')
Path(board_dir).mkdir(parents=True, exist_ok=True)
depth_dir = os.path.join(save_dir, 'depth')
Path(depth_dir).mkdir(parents=True, exist_ok=True)
mask_dir = os.path.join(save_dir, 'mask')
Path(mask_dir).mkdir(parents=True, exist_ok=True)
img_dir = os.path.join(save_dir, 'images')
Path(img_dir).mkdir(parents=True, exist_ok=True)

# setup logger
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("DPA")
logger.setLevel(getattr(logging, args.log_level.upper()))
fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
# fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
# fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# the norm should be aligned with the stereo model's requirement
if args.model == 'raft':
    assert args.img_norm_min == 0 and args.img_norm_max == 255, "the norm should be [0, 1] for RaftStereo model"
else:
    assert args.img_norm_min == 0 and args.img_norm_max == 1, "the norm should be [0, 255] for all models except RaftStereo"

# create & load the depth estimation model
model = StereoModel(method=args.model, device=device)
model.restore_model(args.ckpt)
model.eval()
logger.info("Model Loaded Successfully!")

# get the pixel position and size of the board in the scene
pos_left, pos_right, boardL_size, boardR_size = get_board_spec(phy_size=(0.891,1.26), phy_depth=5, center_shift=(0,0))
# round the specification numbers
pos_left, pos_right = round_numbers(*pos_left), round_numbers(*pos_right)
board_size = round_numbers(*boardL_size) # taking the left board size as anchor
mask_size = (round(boardL_size[0]/args.n_ytile), round(boardL_size[1]/args.n_xtile))
logger.debug("The rounded board specification:")
logger.debug("pos_left --> %s", str(pos_left))
logger.debug("pos_right --> %s", str(pos_right))
logger.debug("board_size --> %s", str(board_size))
logger.debug("mask_size --> %s", str(mask_size))

# the board ground truth depth is dataset-specific
phy_disp = 0.54 * 721 / 5 # KITTI dataset


# create the optimizable mask
mask_factory = DPABoard(
    mask_size=mask_size, 
    board_size=board_size, 
    colored=args.colored,
    device=device)

unit_optimizer = torch.optim.Adam([mask_factory._unit], lr=0.01)

assert args.grad_step_interval <= args.round, "the gradient update interval must be smaller than the total iteration counts"
mask_grad_sch = Scheduler(
    mode='linear',
    start=args.grad_norm_start, 
    end=args.grad_norm_end, 
    n_step=args.round//args.grad_step_interval
)
assert args.binary_step_interval <= args.round, "the gradient update interval must be smaller than the total iteration counts"
unit_binary_sch = Scheduler(
    mode='linear',
    start=args.bin_norm_start, 
    end=args.bin_norm_end, 
    n_step=args.round//args.binary_step_interval
)

# torch.autograd.set_detect_anomaly(True)
logger.info("Start Optimizing the Mask!\n")

for rnd in range(args.round):
    logger.info("\nOptimiztion Round {:03d}".format(rnd))

    logger.debug("step [1]: load the stereo pair")
    # load the stereo pair
    img_left, img_right = load_image(args.img_left, expand=True), load_image(args.img_right, expand=True)
    # normalization
    img_left, img_right = StereoDataset.normalize(
        [img_left, img_right],
        [args.img_norm_min, args.img_norm_max]
    )
    img_left, img_right = img_left.to(device), img_right.to(device)

    gt_disp = torch.full([1,1] + list(img_left.size()[-2:]), fill_value=phy_disp, device=device)
    if args.model.lower() != 'raftstereo': # insufficient GPU memory for RAFT-Stereo
        pseudo_disp = model.forward(img_left, img_right)
    else:
        pseudo_disp = None

    logger.debug("step [2]: assemble the board as tiled masks")
    opt_board = mask_factory.assemble_opt_board()

    logger.debug("step [3]: embed board into the stereo images")
    # assemble the optimizable stereo images
    imgL, imgR = embed_board_into_stereo(
        opt_board, 
        img_left, img_right, 
        pos_left, pos_right, 
        board_size,
        [args.img_norm_min, args.img_norm_max],
        tau=args.relax_temp
    )

    logger.debug("step [4]: estimate the disparity of the perturbed stereo images")
    # get estimation result
    perturb_disp = model.forward(imgL, imgR)

    logger.debug("step [5]: calculate the loss for the board")
    # calculate loss: mse, smooth, nps
    loss = regional_mean_square_error(perturb_disp, tl_con=pos_left[0], board_size=board_size) + args.alpha*entropy_loss(opt_board) + args.gamma*tv_loss(opt_board)
    # + args.beta*contrast_loss(opt_board)  + args.delta*frequency_loss(opt_mask)

    logger.debug("step [6]: backward the loss with respect to the board")
    # board_grad = autograd.grad(loss, opt_board)[0]
    unit_optimizer.zero_grad()
    loss.backward()
    unit_optimizer.step()
    mask_factory._unit.data = lp_projection(mask_factory._unit.data, 0.5, 'inf')
    # print(mask_factory._unit.min(), mask_factory._unit.max())
    # xxx

    # logger.debug("step [7]: aggregate the gradients")
    # if rnd % args.grad_step_interval == 0:
    #     grad_cst = mask_grad_sch.next()
    #     logger.info("Step the mask gradient constraint as %s", str(grad_cst))
    # mask_grad = aggregate_board_grad(
    #     board_grad,
    #     mask_size, 
    #     colored=args.colored,
    #     constraint=None,
    # )

    if rnd % args.n_checkpoint == 0:
        logger.info("Save the Progress of the Optimization: %s\n", save_dir)

        # save the perturbed stereo images
        save_img(
            imgL,
            os.path.join(img_dir, 'round_{:03d}_left.jpg'.format(rnd))
        )
        save_img(
            imgR,
            os.path.join(img_dir, 'round_{:03d}_right.jpg'.format(rnd))
        )

        # save the mask unit
        save_perturb(
            mask_factory._unit,
            os.path.join(mask_dir, 'round_{:03d}.jpg'.format(rnd))
        )

        # save the assembled board
        save_perturb(
            opt_board,
            os.path.join(board_dir, 'round_{:03d}.jpg'.format(rnd))
        )

        # save the perturbed depth
        save_depth(
            perturb_disp,
            os.path.join(depth_dir, 'round_{:03d}_perturb.jpg'.format(rnd)),
            # max_disp=150,
            cm=cv2.COLORMAP_JET,
            invert=True
        )
        # save the clean depth
        if pseudo_disp is not None:
            save_depth(
                pseudo_disp,
                os.path.join(depth_dir, 'round_{:03d}_clean.jpg'.format(rnd)),
                # max_disp=150,
                cm=cv2.COLORMAP_JET,
                invert=True
            )
    
    # logger.debug("step [8]: update the board unit with annealing")
    # if rnd % args.binary_step_interval == 0:
    #     anneal_cst = unit_binary_sch.next()
    #     logger.info("Step the board unit annealing constraint as %s", str(anneal_cst))
    # # update after save 
    # mask_factory.update_mask(mask_grad, anneal_para=None)

    logger.info("Optimization Done: loss {:05f}\n".format(loss))