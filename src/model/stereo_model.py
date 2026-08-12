# -*- encoding: utf-8 -*-
"""
@File    :   stereo_model.py
@Time    :   2025/04/28 21:25:33
@Author  :   yxing
"""

import torch

class StereoModel(object):
    '''
    Wrapper class for all stereo models

    Arg(s):
        method : str
            stereo model to use
        num_deform_layers : int
            number of deformable convolution layers
        device : torch.device
            device to run optimization
    '''

    def __init__(self,
                 method,
                 num_deform_layers=0,
                 device='cpu'):

        self.method = method

        self.num_deform_layers = num_deform_layers

        if method == 'psmnet':
            from .psmnet_model import PSMNetModel
            self.model = PSMNetModel(
                num_deform_layers=self.num_deform_layers,
                device=device)
        elif method == 'deeppruner':
            from .deeppruner_model import DeepPrunerModel
            self.model = DeepPrunerModel(
                method='best',
                num_deform_layers=self.num_deform_layers,
                device=device)
        elif method == 'aanet':
            from .aanet_model import AANetModel
            self.model = AANetModel(
                variant='regular',
                device=device)
        elif method == 'raftstereo':
            from .raftstereo_model import RaftStereoModel
            self.model = RaftStereoModel(
                device=device)
        elif method == 'sttr':
            from .sttr_model import STTRModel
            self.model = STTRModel(
                device=device)
        # elif method == 'igev':
        #     from .igev_model import IGEVStereoModel
        #     self.model = IGEVStereoModel(
        #         device=device)
        # elif method == 'gmstereo':
        #     from .gmstereo_model import GMStereoModel
        #     self.model = GMStereoModel(
        #         device=device)
        else:
            raise ValueError('Unsupported stereo model: {}'.format(method))

    def forward(self, image0, image1):
        '''
        Forwards stereo pair through network

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W left image
            image1 : torch.Tensor[float32]
                N x C x H x W right image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W disparity
        '''

        outputs = self.model.forward(image0, image1)
        return outputs

    def compute_loss(self, outputs, ground_truth, pseudo_ground_truth=None, **kwargs):
        '''
        Computes training loss

        Arg(s):
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W disparity
            outputs : list[torch.Tensor[float32]]
                list of N x 1 x H x W disparity
            pseudo_ground_truth : torch.Tensor[float32]
                N x 1 x H x W disparity
        Returns:
            float : loss
        '''
        regional = True if 'region_tl' in kwargs and 'region_size' in kwargs else False

        if self.method == 'aanet' or self.method == 'aanet_plus':
            if regional:
                return self.model.compute_regional_loss(
                    outputs=outputs,
                    ground_truth=ground_truth,
                    region_tl=kwargs['region_tl'],
                    region_size=kwargs['region_size'],
                    pseudo_ground_truth=pseudo_ground_truth)
            else:
                return self.model.compute_loss(
                    outputs=outputs,
                    ground_truth=ground_truth,
                    pseudo_ground_truth=pseudo_ground_truth)
        else:
            if regional:
                return self.model.compute_regional_loss(
                    outputs=outputs,
                    ground_truth=ground_truth,
                    region_tl=kwargs['region_tl'],
                    region_size=kwargs['region_size'])
            else:
                return self.model.compute_loss(
                    outputs=outputs,
                    ground_truth=ground_truth)

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

        return self.model.train(flag_only=flag_only)

    def eval(self, flag_only=False):
        '''
        Sets model to evaluation mode

        Arg(s):
            flag_only : bool
                if set, then only sets the eval flag, but not mode
        '''

        return self.model.eval(flag_only=flag_only)

    def save_model(self, save_path):
        '''
        Stores weights into a checkpoint

        Arg(s):
            save_path : str
                path to model weights
        '''

        self.model.save_model(save_path)

    def restore_model(self, restore_path):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
        '''

        self.model.restore_model(restore_path)
