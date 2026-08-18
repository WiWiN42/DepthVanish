# -*- encoding: utf-8 -*-
"""
@File    :   aanet_model.py
@Time    :   2025/04/28 21:26:23
@Author  :   yxing
"""


import os, sys
import torch, torchvision
from argparse import Namespace
# Use this file's own directory rather than the process's cwd -- raft_stereo.py does absolute
# imports like `from core.update import ...` that require RAFT-Stereo's own subdirectories on
# sys.path, regardless of where this script was invoked from.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(_MODEL_DIR, 'RAFT-Stereo'))
sys.path.insert(0, os.path.join(_MODEL_DIR, 'RAFT-Stereo', 'core'))
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder


class RaftStereoModel(object):
    '''
    Wrapper class for RAFT-Stereo model

    Arg(s):
        variant : str
            aanet model to use: regular (AANet), plus (AANet+)
        device : torch.device
            cpu or cuda device to run on
    '''

    def __init__(self, device=torch.device('cuda')):

        self.max_disparity = 192
        self.device = device

        # create & load the depth estimation model
        self.model_para = Namespace(
            save_numpy=False, 
            # left_imgs='/home/yxing/projects/stereo_PhysicalAttack/cropr1_pasted.png', 
            # right_imgs='/mnt/data/data_yxing/KITTI/training/image_3/000005_10.png', 
            output_directory='demo_output',
            mixed_precision=True,
            valid_iters=25,
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

        self.model = torch.nn.DataParallel(RAFTStereo(self.model_para), device_ids=[self.device])

        # Move to device
        self.to(self.device)
        self.eval()

    def forward(self, image0, image1):
        '''
        Forwards stereo pair through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W disparity if mode is 'eval'
            list[torch.Tensor[float32]] : N x 1 x H x W disparity if mode is 'train'
        '''
        # the input is normalized into [0,1], but raft-stereo requires [0,255]
        image0, image1 = image0 * 255.0, image1 * 255.0

        padder = InputPadder(image0.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image0, image1)

        _, flow_up = self.model(image1_pad, image2_pad, iters=self.model_para.valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()
        pred_disp = -flow_up.squeeze()

        return pred_disp

    def transform_inputs(self, image0, image1):
        '''
        Transforms the stereo pair using standard normalization as a preprocessing step

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
        Returns:
            torch.Tensor[float32] : N x 3 x H x W left image
            torch.Tensor[float32] : N x 3 x H x W right image
        '''

        normal_mean_var = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

        transform_func = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(**normal_mean_var)])

        n_batch, _, n_height, n_width = image0.shape

        image0 = torch.chunk(image0, chunks=n_batch, dim=0)
        image1 = torch.chunk(image1, chunks=n_batch, dim=0)

        image0 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image0
        ], dim=0)
        image1 = torch.stack([
            transform_func(torch.squeeze(image)) for image in image1
        ], dim=0)

        if self.variant == 'regular':
            downsample_scale = 12
        elif self.variant == 'plus':
            downsample_scale = 32
        else:
            raise NotImplementedError('Specified AANet variant not implemented: {}'.format(self.variant))

        # Pad images along top and right dimensions
        padding_top = int(downsample_scale - (n_height % downsample_scale))
        padding_right = int(downsample_scale - (n_width % downsample_scale))

        image0 = torch.nn.functional.pad(
            image0,
            (0, padding_right, padding_top, 0, 0, 0),
            mode='constant',
            value=0)
        image1 = torch.nn.functional.pad(
            image1,
            (0, padding_right, padding_top, 0, 0, 0),
            mode='constant',
            value=0)

        return image0, image1, padding_top, padding_right

    def compute_loss(self, outputs, ground_truth, pseudo_ground_truth=None):
        '''
        Computes training loss

        Arg(s):
            outputs : list[torch.Tensor[float32]]
                list of N x 1 x H x W output disparity
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W  disparity
            pseudo_ground_truth : torch.Tensor[float32]
                N x 1 x H x W  disparity
        Returns:
            float : loss
        '''

        mask_ground_truth = \
            (ground_truth > 0) & (ground_truth < self.max_disparity)

        mask_ground_truth.detach_()

        output = outputs[-1]

        # Select outputs where disparity is defined
        loss = torch.nn.functional.smooth_l1_loss(
            output[mask_ground_truth],
            ground_truth[mask_ground_truth],
            reduction='mean')

        if pseudo_ground_truth is not None and torch.max(pseudo_ground_truth) > 0:

            mask_pseudo_ground_truth = ground_truth <= 0

            mask_pseudo_ground_truth.detach_()

            # Compute loss with pseudo groundtruth where groundtruth is not available
            loss_pseudo_ground_truth = torch.nn.functional.smooth_l1_loss(
                output[mask_pseudo_ground_truth],
                pseudo_ground_truth[mask_pseudo_ground_truth],
                reduction='mean')

            loss = loss + loss_pseudo_ground_truth

        return loss

    def compute_regional_loss(self, outputs, ground_truth, region_tl, region_size, pseudo_ground_truth=None):
        
        start_x, end_x  = region_tl[0], region_tl[0] + region_size[1]
        start_y, end_y  = region_tl[1], region_tl[1] + region_size[0]

        output = outputs[-1]
        output = output[..., start_y:end_y, start_x:end_x]
        ground_truth = ground_truth[..., start_y:end_y, start_x:end_x]
        if pseudo_ground_truth is not None:
            pseudo_ground_truth = pseudo_ground_truth[..., start_y:end_y, start_x:end_x]

        mask_ground_truth = \
            (ground_truth > 0) & (ground_truth < self.max_disparity)

        mask_ground_truth.detach_()

        # Select outputs where disparity is defined
        loss = torch.nn.functional.smooth_l1_loss(
            output[mask_ground_truth],
            ground_truth[mask_ground_truth],
            reduction='mean')

        if pseudo_ground_truth is not None and torch.max(pseudo_ground_truth) > 0:

            mask_pseudo_ground_truth = ground_truth <= 0

            mask_pseudo_ground_truth.detach_()

            # Compute loss with pseudo groundtruth where groundtruth is not available
            loss_pseudo_ground_truth = torch.nn.functional.smooth_l1_loss(
                output[mask_pseudo_ground_truth],
                pseudo_ground_truth[mask_pseudo_ground_truth],
                reduction='mean')

            loss = loss + loss_pseudo_ground_truth

        return loss

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

    def named_parameters(self):
        '''
        Returns the list of named parameters in the model

        Returns:
            dict[str, torch.Tensor[float32]] : list of parameters
        '''

        return self.model.named_parameters()

    def train(self, flag_only=False):
        '''
        Sets model to training mode

        Arg(s):
            flag_only : bool
                if set, then only sets the train flag, but not mode
        '''

        if not flag_only:
            self.model.train()

        self.mode = 'train'

    def eval(self, flag_only=False):
        '''
        Sets model to evaluation mode

        Arg(s):
            flag_only : bool
                if set, then only sets the eval flag, but not mode
        '''

        if not flag_only:
            self.model.eval()

        self.mode = 'eval'

    def to(self, device):
        '''
        Moves model to device

        Arg(s):
            device : torch.device
                cpu or cuda device to run on
        '''

        # Move to device
        self.model.to(device)

    def save_model(self, save_path):
        '''
        Stores weights into a checkpoint

        Arg(s):
            save_path : str
                path to model weights
        '''

        checkpoint = {
            'state_dict' : self.model.state_dict()
        }

        torch.save(checkpoint, save_path)

    def restore_model(self, restore_path):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
        '''

        checkpoint = torch.load(restore_path)
        self.model.load_state_dict(checkpoint, strict=True)
