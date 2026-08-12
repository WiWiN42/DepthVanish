# -*- encoding: utf-8 -*-
"""
@File    :   sttr_model.py
@Time    :   2025/04/28 21:26:23
@Author  :   yxing
"""

import os, sys
import torch, torchvision
from argparse import Namespace

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join('model', 'stereo-transformer'))
from module.sttr import STTR
from utilities.misc import NestedTensor


class STTRModel(object):
    '''
    Wrapper class for STTR (Stereo Transformer) model

    Arg(s):
        device : torch.device
            cpu or cuda device to run on
    '''

    def __init__(self, device=torch.device('cuda')):

        self.max_disparity = 192
        self.device = device

        # Build STTR with default args matching the pretrained KITTI checkpoint
        args = Namespace(
            channel_dim=128,
            position_encoding='sine1d_rel',
            num_attn_layers=6,
            nheads=8,
            regression_head='ot',
            context_adjustment_layer='cal',
            cal_num_blocks=8,
            cal_feat_dim=16,
            cal_expansion_ratio=4,
        )

        self.model = STTR(args)

        # Move to device
        self.to(self.device)
        self.eval()

    def forward(self, image0, image1):
        '''
        Forwards stereo pair through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image  (values in [0, 1])
            image1 : torch.Tensor[float32]
                N x C x H x W right image (values in [0, 1])
        Returns:
            torch.Tensor[float32] : N x 1 x H x W disparity
        '''

        image0, image1 = self.transform_inputs(image0, image1)

        inputs = NestedTensor(image0, image1, sampled_cols=None, sampled_rows=None)

        outputs = self.model(inputs)

        # disp_pred is [N, H, W]; unsqueeze to [N, 1, H, W]
        disp_pred = outputs['disp_pred'].unsqueeze(1)

        return disp_pred

    def transform_inputs(self, image0, image1):
        '''
        Applies ImageNet normalisation to the stereo pair

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
        Returns:
            torch.Tensor[float32] : N x 3 x H x W normalised left image
            torch.Tensor[float32] : N x 3 x H x W normalised right image
        '''

        normal_mean_var = {
            'mean': [0.485, 0.456, 0.406],
            'std':  [0.229, 0.224, 0.225]
        }

        transform_func = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(**normal_mean_var)])

        n_batch = image0.shape[0]

        image0 = torch.stack([
            transform_func(image0[i]) for i in range(n_batch)
        ], dim=0)
        image1 = torch.stack([
            transform_func(image1[i]) for i in range(n_batch)
        ], dim=0)

        return image0, image1

    def compute_loss(self, outputs, ground_truth):
        '''
        Computes training loss

        Arg(s):
            outputs : torch.Tensor[float32]
                N x 1 x H x W predicted disparity
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground-truth disparity
        Returns:
            float : loss
        '''

        mask = (ground_truth > 0) & (ground_truth < self.max_disparity)
        mask.detach_()

        loss = torch.nn.functional.smooth_l1_loss(
            outputs[mask],
            ground_truth[mask],
            reduction='mean')

        return loss

    def compute_regional_loss(self, outputs, ground_truth, region_tl, region_size):
        '''
        Computes training loss over a spatial region

        Arg(s):
            outputs : torch.Tensor[float32]
                N x 1 x H x W predicted disparity
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground-truth disparity
            region_tl : tuple[int, int]
                (x, y) top-left corner of the region
            region_size : tuple[int, int]
                (height, width) of the region
        Returns:
            float : loss
        '''

        start_x, end_x = region_tl[0], region_tl[0] + region_size[1]
        start_y, end_y = region_tl[1], region_tl[1] + region_size[0]

        outputs      = outputs[...,      start_y:end_y, start_x:end_x]
        ground_truth = ground_truth[..., start_y:end_y, start_x:end_x]

        return self.compute_loss(outputs, ground_truth)

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
            dict[str, torch.Tensor[float32]] : name, parameters pair
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

        self.model.to(device)

    def save_model(self, save_path):
        '''
        Stores weights into a checkpoint

        Arg(s):
            save_path : str
                path to model weights
        '''

        checkpoint = {
            'state_dict': self.model.state_dict()
        }

        torch.save(checkpoint, save_path)

    def restore_model(self, restore_path):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
        '''

        checkpoint = torch.load(restore_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        missing, unexpected = self.model.load_state_dict(pretrained_dict, strict=False)

        if missing:
            raise RuntimeError('Missing keys in checkpoint: {}'.format(', '.join(missing)))

        # BN running stats may appear as unexpected — that is acceptable
        unexpected_filtered = [
            k for k in unexpected
            if 'running_mean' not in k and 'running_var' not in k
        ]
        if unexpected_filtered:
            raise RuntimeError('Unexpected keys in checkpoint: {}'.format(', '.join(unexpected_filtered)))