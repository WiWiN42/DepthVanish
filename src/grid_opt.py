# -*- encoding: utf-8 -*-
"""
@File    :   grid_opt.py
@Time    :   2025/05/13 11:07:30
@Author  :   yxing
"""


import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import cv2

from utils.assemble import DPABoard
from utils.dataset import StereoDataset
from utils.deploy import DigitalDeploy
from model.stereo_model import StereoModel
from utils.metric import cal_metric
from utils.tool import save_img, save_depth, save_perturb, load_image, lp_projection, round_numbers
from utils.loss import regional_mean_square_error, entropy_loss, tv_loss

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="test", help="the exp name")
parser.add_argument("--round", type=int, default=10, help="the number of rounds for searching an effective patch")
parser.add_argument("--n_checkpoint", type=int, default=20, help="the save interval")
parser.add_argument("--save_dir", type=str, default='./result/', help="the directory to save middle results")
parser.add_argument("--log_level", type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help="the log level for python logger")

# patch settings
parser.add_argument("--colored", action='store_true', help="whether set the mask as 3 channel colored or not")
parser.add_argument("--n_xtile", type=int, default=5, help="the number of horizontal tile to get the patch")
parser.add_argument("--n_ytile", type=int, default=5, help="the number of horizontal tile to get the patch")
parser.add_argument("--physical_height", type=float, help="the physical height (meter) of the patch in the scene")
parser.add_argument("--physical_width", type=float, help="the physical width (meter) of the patch in the scene")
parser.add_argument("--physical_depth", type=float, help="the physical depth (meter) of the patch in the scene")
parser.add_argument("--physical_shiftx", type=float, help="the physical distance (meter) to shift the patch along x axis")
parser.add_argument("--physical_shifty", type=float, help="the physical distance (meter) to shift the patch along y axis")

# optimization settings
parser.add_argument("--unit_norm", type=float, default=0.5, help="the range to constraint the unit value")
parser.add_argument("--alpha", type=float, default=1, help="the weighting parameter of regional MSE of the patch")
parser.add_argument("--beta", type=float, default=1, help="the weighting parameter of enhancing mask sharpness")
parser.add_argument("--gamma", type=float, default=1, help="the weighting parameter of encouraging smooth areas")
parser.add_argument("--delta", type=float, default=1, help="the weighting parameter of encouraging smooth areas")

# model settings
parser.add_argument("--model", type=str, default='raft', help="the model to adopt")
parser.add_argument("--ckpt", type=str, help="the pre-trained depth estimation model")

# image normalization range
parser.add_argument("--img_norm_min", type=int, default=0, help="the minimum value to normalize the stereo images")
parser.add_argument("--img_norm_max", type=int, default=1, help="the maximum value to normalize the stereo images")

# stereo pair data
parser.add_argument("--dataset", type=str, help="the dataset name")
parser.add_argument("--img_left", type=str, default="/mnt/data/data_yxing/KITTI_stereo_2015/training/image_2/000003_10.png", help="the left image of stereo pair to demo")
parser.add_argument("--img_right", type=str, default="/mnt/data/data_yxing/KITTI_stereo_2015/training/image_3/000003_10.png", help="the right image of stereo pair to demo")
parser.add_argument("--stereo_calib", type=str, default="/mnt/data/data_yxing/KITTI_stereo_2015/training/calib_cam_to_cam/000003.txt", help="the corresponding calibration file")

args = parser.parse_args()


# setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# setup save root first
save_dir = os.path.join(os.getcwd(), args.save_dir, args.exp)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
patch_dir = os.path.join(save_dir, 'patch')
Path(patch_dir).mkdir(parents=True, exist_ok=True)
depth_dir = os.path.join(save_dir, 'depth')
Path(depth_dir).mkdir(parents=True, exist_ok=True)
unit_dir = os.path.join(save_dir, 'unit')
Path(unit_dir).mkdir(parents=True, exist_ok=True)
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

# create & load the depth estimation model
model = StereoModel(method=args.model, device=device)
model.restore_model(args.ckpt)
model.eval()
logger.info("Model Loaded Successfully!")

# digital deployment utility
digi_deployer = DigitalDeploy(
    args.dataset.lower(), 
    args.stereo_calib, 
    args.img_left, args.img_right, 
)

# get the pixel position and size of the patch in the scene
pos_left, pos_right, patchL_size, patchR_size = digi_deployer.get_patch_spec( 
    physical_size=(args.physical_height, args.physical_width), 
    physical_depth=args.physical_depth,
    center_shift=(args.physical_shiftx, args.physical_shifty)
)
# round the specification numbers
pos_left, pos_right = round_numbers(*pos_left), round_numbers(*pos_right)
patch_size = round_numbers(*patchL_size) # taking the left patch size as anchor
# determine the unit size according the interval spec (3x4)
unit_h, unit_w = (patch_size[0]-3*10)//4, (patch_size[1]-4*10)//5
unit_size = (unit_h, unit_w)
logger.info("The rounded patch specification:")
logger.info("pos_left --> %s", str(pos_left))
logger.info("pos_right --> %s", str(pos_right))
logger.info("patch_size --> %s", str(patch_size))
logger.info("unit_size --> %s", str(unit_size))

# the patch ground truth depth is dataset-specific
if args.dataset.lower() == 'kitti':
    phy_disp = 0.54 * 721 / args.physical_depth
elif args.dataset.lower() == 'drivingstereo':
    phy_disp = 0.54 * 200 / args.physical_depth
else:
    raise ValueError("unknown dataset")
patch_gt_disp = torch.full(list(patch_size), fill_value=phy_disp, device=device)


# create the optimizable mask
patch_manager = DPABoard(
    mask_size=unit_size, 
    board_size=patch_size, 
    colored=args.colored,
    device=device
)

unit_optimizer = torch.optim.Adam([patch_manager._unit], lr=0.01)

# torch.autograd.set_detect_anomaly(True)
logger.info("Start Optimizing the patch!\n")

best_result = dict(
    round=0,
    d1=0,
    epe=0
)
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

    logger.debug("step [2]: assemble the patch as tiled unit")
    opt_board = patch_manager.assemble_grid_board()

    logger.debug("step [3]: embed board into the stereo images")
    # assemble the optimizable stereo images
    imgL, imgR = digi_deployer.deploy(
        opt_board, 
        img_left, img_right, 
        pos_left, pos_right, 
        patch_size,
    )

    logger.debug("step [4]: estimate the disparity of the perturbed stereo images")
    # get estimation result
    perturb_disp = model.forward(imgL, imgR)

    logger.debug("step [5]: calculate the loss for the patch")
    # calculate loss: mse, smooth, nps
    loss = regional_mean_square_error(perturb_disp, tl_con=pos_left[0], board_size=patch_size)
    # + args.beta*contrast_loss(opt_board)  + args.delta*frequency_loss(opt_mask)

    logger.debug("step [6]: backward the loss with respect to the board")
    # board_grad = autograd.grad(loss, opt_board)[0]
    unit_optimizer.zero_grad()
    loss.backward()
    unit_optimizer.step()

    start_x, end_x  = pos_left[0][0], pos_left[0][0] + patch_size[1]
    start_y, end_y  = pos_left[0][1], pos_left[0][1] + patch_size[0]
    patch_perturb_disp = perturb_disp[..., start_y:end_y, start_x:end_x]

    d1_error, epe = cal_metric(patch_perturb_disp, patch_gt_disp)

    logger.info('{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('', 'D1-Error ', '+/-', 'EPE', '+/-'))
    logger.info('{:<10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
        '',
        d1_error,
        0,
        epe,
        0,
    ))

    if rnd % args.n_checkpoint == 0:
        logger.info("Save the Progress of the Optimization: %s\n", save_dir)

        # update the best results first
        if d1_error > best_result['d1']:
            best_result['round'] = rnd
            best_result['d1'] = d1_error
            best_result['epe'] = epe

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
            patch_manager._unit,
            os.path.join(unit_dir, 'round_{:03d}.jpg'.format(rnd))
        )

        # save the assembled board
        save_perturb(
            opt_board,
            os.path.join(patch_dir, 'round_{:03d}.jpg'.format(rnd))
        )

        # save the perturbed depth
        save_depth(
            perturb_disp,
            os.path.join(depth_dir, 'round_{:03d}_perturb.jpg'.format(rnd)),
            # max_disp=150,
            cm=cv2.COLORMAP_JET,
            invert=True
        )

    logger.info("Optimization Done: loss {:05f}\n".format(loss))

logger.info('The best result (D1-error Anchored):')
logger.info('{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('', 'D1-Error ', '+/-', 'EPE', '+/-'))
logger.info('{:<10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
    '',
    d1_error,
    0,
    epe,
    0,
))