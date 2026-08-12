# -*- encoding: utf-8 -*-
"""
@File    :   eval_dataset.py
@Time    :   2025/05/09 18:07:25
@Author  :   yxing
"""


import os
from argparse import Namespace

import cv2
import torch
import numpy as np
from torchvision.transforms.functional import resize as tt_resize

from utils.deploy import DigitalDeploy
from utils.dataset import StereoDataset
from utils.metric import d1_error, end_point_error, cal_metric
from model.stereo_model import StereoModel
from utils.tool import save_img, save_depth, load_image, round_numbers, read_paths


# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/pretrained_model_KITTI2015.tar',

# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/DeepPruner-best-kitti.tar',

# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/aanet_kitti15-fb2a0d23.pth',

# ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/raftstereo-sceneflow.pth',

args = Namespace(
    # model='psmnet',
    # ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/pretrained_model_KITTI2015.tar',
    # model='deeppruner',
    # ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/DeepPruner-best-kitti.tar',
    model='aanet',
    ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/aanet_kitti15-fb2a0d23.pth',
    # model='raftstereo',
    # ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/raftstereo-sceneflow.pth',
    img_norm_min=0,
    img_norm_max=1,
    save_dir='result/eval_dataset/aanet_cloudy_stereoscopic',
    img_patch='/home/yxing/projects/stereo_PhysicalAttack/data/stereoscopic.png',
    # img_patch='/home/yxing/projects/stereo_PhysicalAttack/data/stereopagnosia.png',
    physical_size=(0.891,1.26),
    physical_depth=5,
    physical_center_shift=(0,0),
    # dataset='KITTI',
    # img_left_file='/home/yxing/projects/stereo_PhysicalAttack/data/kitti_scene_flow_val_image0.txt',
    # img_right_file='/home/yxing/projects/stereo_PhysicalAttack/data/kitti_scene_flow_val_image1.txt',
    dataset='DrivingStereo',
    img_left_file='/home/yxing/projects/stereo_PhysicalAttack/data/drivingstereo_cloudy_val_image0.txt',
    img_right_file='/home/yxing/projects/stereo_PhysicalAttack/data/drivingstereo_cloudy_val_image1.txt',
    rescale=1
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

# load all the data
left_images, right_images = read_paths(args.img_left_file), read_paths(args.img_right_file)

all_d1_result = []
all_epe_result = []

for img_left_file, img_right_file in zip(left_images, right_images):

    print("Start evaluation for {}".format(img_left_file))
    img_identifier = img_left_file.split('/')[-1].split('.')[0]

    # find the corresponding calibration file first
    calib_name = img_left_file.split('/')[-1].split('_')[0] + '.txt'
    if args.dataset.lower() == 'kitti':
        stereo_calib = os.path.join("/mnt/data/data_yxing/KITTI_stereo_2015/training/calib_cam_to_cam", calib_name)
    elif args.dataset.lower() == 'drivingstereo':
        stereo_calib = os.path.join("/mnt/data/data_yxing/DrivingStereo/full-image-calib", calib_name)
    else:
        raise ValueError("unknown dataset")

    # digital deployment utility
    digi_deploy = DigitalDeploy(
        args.dataset.lower(), 
        stereo_calib, 
        img_left_file, img_right_file, 
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
    patch_gt_disp = torch.full(list(patch_size), fill_value=phy_disp)

    # load patch
    patch = load_image(args.img_patch, size=(patch_size[1], patch_size[0]))

    # load the stereo pair
    img_left, img_right = load_image(img_left_file, expand=True), load_image(img_right_file, expand=True)
    # normalization
    img_left, img_right, patch = StereoDataset.normalize(
        [img_left, img_right, patch],
        [args.img_norm_min, args.img_norm_max]
    )

    # past the patch onto the stereo image
    imgL, imgR = digi_deploy.deploy(patch, img_left, img_right, pos_left, pos_right, patch_size)

    # resize the image tensor to avoid cuda out-of-memory
    assert args.rescale > 0
    original_size = tuple(img_left.size()[-2:])
    target_size = (round(original_size[0]*args.rescale), round(original_size[1]*args.rescale))

    # forward to get disparity prediction
    pred_disp = model.forward(
        tt_resize(imgL, target_size).to(device), 
        tt_resize(imgR, target_size).to(device)
    )

    # resize back the prediction
    if args.model == 'raftstereo':
        pred_disp = tt_resize(pred_disp.unsqueeze(0), original_size).squeeze()
    else:
        pred_disp = tt_resize(pred_disp, original_size).squeeze()

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
    all_d1_result.append(d1)
    all_epe_result.append(epe)

    # save the perturbed stereo images
    save_img(
        imgL,
        os.path.join(save_dir, img_identifier+'_left_img.png')
    )
    # save the perturbed depth
    save_depth(
        pred_disp,
        os.path.join(save_dir, img_identifier+'_depth_map.png'),
        # max_disp=150,
        cm=cv2.COLORMAP_JET,
        invert=True
    )
print('=======================================')
print('The overall results are:')
print('{:<10}  {:>10}  {:>10}  {:>10}  {:>10}'.format('', 'D1-Error ', '+/-', 'EPE', '+/-'))
print('{:<10}  {:>10.4f}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
    '',
    np.mean(all_d1_result)*100.0,
    np.std(all_d1_result)*100.0,
    np.mean(all_epe_result),
    np.std(all_epe_result),
))