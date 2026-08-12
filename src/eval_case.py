# -*- encoding: utf-8 -*-
"""
@File    :   eval_case.py
@Time    :   2025/05/09 10:44:43
@Author  :   yxing
"""


import os
from argparse import Namespace

import torch
import cv2
from torchvision.transforms.functional import resize as tt_resize

from utils.deploy import DigitalDeploy
from model.stereo_model import StereoModel
from utils.dataset import StereoDataset
from utils.metric import d1_error, end_point_error, cal_metric
from utils.tool import save_img, save_depth, load_image, round_numbers


# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/pretrained_model_KITTI2015.tar',
# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/DeepPruner-best-kitti.tar',
# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/aanet_kitti15-fb2a0d23.pth',
# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/raftstereo-sceneflow.pth',

args = Namespace(
    model='raft',
    ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/aanet_kitti15-fb2a0d23.pth',
    dataset='drivingstereo',
    img_norm_min=0,
    img_norm_max=1,
    run_clean=True, # get original depth results if True
    save_dir='result/case_test',
    img_patch='/home/yxing/projects/stereo_PhysicalAttack/assets/patches/aanet.jpg',
    physical_size=(0.891,1.26),
    physical_depth=5,
    physical_center_shift=(0,0),
    # img_left='/mnt/data/data_yxing/DrivingStereo/cloudy/data/left-image-full-size/2018-10-31-06-55-01_2018-10-31-06-55-02-084.png',
    # img_right='/mnt/data/data_yxing/DrivingStereo/cloudy/data/right-image-full-size/2018-10-31-06-55-01_2018-10-31-06-55-02-084.png',
    stereo_calib='/mnt/data/data_yxing/DrivingStereo/full-image-calib/2018-10-31-06-55-01.txt',
    img_left='/home/yxing/projects/stereo_PhysicalAttack/assets/physical/1-3_left.jpg',
    img_right='/home/yxing/projects/stereo_PhysicalAttack/assets/physical/1-3_right.jpg',
    # stereo_calib='/mnt/data/data_yxing/KITTI_stereo_2015/training/calib_cam_to_cam/000003.txt',
    rescale=1 # the scale to resize the image to run
)

# setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# setup save root first
save_dir = os.path.join(os.getcwd(), args.save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# create & load the depth estimation model
model = StereoModel(method=args.model, device=device)
model.restore_model(args.ckpt)
model.eval()

# deploy the patch into stereo frames
digi_deploy = DigitalDeploy(
    args.dataset.lower(), 
    args.stereo_calib, 
    args.img_left, args.img_right, 
)

# get the pixel position and size of the patch in the scene
pos_left, pos_right, patchL_size, patchR_size = digi_deploy.get_patch_spec(
    physical_size=args.physical_size, 
    physical_depth=args.physical_depth,
    center_shift=args.physical_center_shift
)
    
# round the specification numbers
pos_left, pos_right = round_numbers(*pos_left), round_numbers(*pos_right)
patch_size = round_numbers(*patchL_size) # taking the left patch size as anchor

# the patch ground truth depth is dataset-specific
if args.dataset.lower() == 'kitti':
    phy_disp = 0.54 * 721 / args.physical_depth
elif args.dataset.lower() == 'drivingstereo':
    phy_disp = 0.54 * 200 / args.physical_depth
else:
    raise ValueError("unknown dataset")
patch_gt_disp = torch.full(list(patch_size), fill_value=phy_disp, device=device)

# load patch
patch = load_image(args.img_patch, size=(patch_size[1], patch_size[0]))

# load the stereo pair
img_left, img_right = load_image(args.img_left, expand=True), load_image(args.img_right, expand=True)
# normalization
img_left, img_right, patch = StereoDataset.normalize(
    [img_left, img_right, patch],
    [args.img_norm_min, args.img_norm_max]
)
img_left, img_right, patch = img_left.to(device), img_right.to(device), patch.to(device)

# past the patch onto the stereo image
imgL, imgR = digi_deploy.deploy(patch, img_left.clone().detach(), img_right.clone().detach(), pos_left, pos_right, patch_size)

# resize image according to the given sacle to fit L40 memory
assert args.rescale > 0
original_size = tuple(img_left.size()[-2:])
target_size = (round(original_size[0]*args.rescale), round(original_size[1]*args.rescale))

if args.run_clean:
    pred_disp = model.forward(
        img_left, img_right
    )
else:
    pred_disp = model.forward(
        imgL,
        imgR
    )

# resize back the prediction
if args.model == 'raftstereo':
    pred_disp = pred_disp.unsqueeze(0)
else:
    pred_disp = pred_disp

# the error calculation
start_x, end_x  = pos_left[0][0], pos_left[0][0] + patch_size[1]
start_y, end_y  = pos_left[0][1], pos_left[0][1] + patch_size[0]
patch_pred_disp = pred_disp[..., start_y:end_y, start_x:end_x]

d1, epe = cal_metric(patch_pred_disp, patch_gt_disp)

print('{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('', 'D1-Error ', '+/-', 'EPE', '+/-'))
print('{:<10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
    '',
    d1,
    0,
    epe,
    0,
))

# save the perturbed stereo images
save_img(
    imgL,
    os.path.join(save_dir, 'left_img.png')
)
# save the perturbed depth
save_depth(
    pred_disp,
    os.path.join(save_dir, 'depth_map.png'),
    # max_disp=150,
    cm=cv2.COLORMAP_JET,
    invert=True
)
