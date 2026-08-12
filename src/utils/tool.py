# -*- encoding: utf-8 -*-
"""
@File    :   util.py
@Time    :   2025/03/31 11:57:09
@Author  :   yxing
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2


def fit_disparity_plane(gt_disp: torch.Tensor, surface_mask: np.ndarray,
                        target_mask: np.ndarray) -> torch.Tensor:
    """Fit a linear disparity plane to the surface region and extrapolate to the target region.

    Fits d(x, y) = a*x + b*y + c via least squares on pixels where ``surface_mask``
    is True, then evaluates the plane at all pixels where ``target_mask`` is True.

    If the target region is fully contained within the surface region, the original
    GT disparity values are returned directly (no fitting needed).

    Args:
        gt_disp: Ground-truth disparity map, shape (H, W).
        surface_mask: Boolean mask of the physical surface region, shape (H, W).
        target_mask: Boolean mask of the full patch region to fill, shape (H, W).

    Returns:
        Disparity map with fitted values at ``target_mask`` pixels and zeros elsewhere,
        same shape and dtype as ``gt_disp``.
    """
    result = torch.zeros_like(gt_disp)

    # If target is fully within the surface, use exact GT values
    if np.all(target_mask <= surface_mask):
        result[target_mask] = gt_disp[target_mask]
        return result

    # Otherwise, fit a linear plane and extrapolate
    sy, sx = np.where(surface_mask)
    sd = gt_disp[surface_mask].numpy()
    A = np.column_stack([sx, sy, np.ones(len(sx))])
    coeffs, _, _, _ = np.linalg.lstsq(A, sd, rcond=None)  # (a, b, c)

    dy, dx = np.where(target_mask)
    fitted_vals = coeffs[0] * dx + coeffs[1] * dy + coeffs[2]

    result[dy, dx] = torch.from_numpy(fitted_vals.astype(np.float32))
    return result

class MinMaxNormalizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        min_val = x.min()
        max_val = x.max()
        ctx.save_for_backward(x, min_val, max_val)
        return (x - min_val) / (max_val - min_val + 1e-6)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        x, min_val, max_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input
    
class SignMaskSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output

def round_numbers(*nums):
    re = ()
    for num in nums:
        if isinstance(num, (tuple, list)):
            re+=(round_numbers(*num),)
        else:
            re+=(round(num),)
    return re

def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            # If there was nothing to read
            if path == '':
                break
            path_list.append(path)

    return path_list

def load_image(img, device='cpu', expand=False, size=None):
    img = Image.open(img).convert("RGB")
    if size is not None:
        img = img.resize(size)
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1) # c, h, w
    img = img.to(device)
    return img[None] if expand else img

def load_disparity(path, multiplier=256.0):
    '''
    Loads a disparity image

    Arg(s):
        path : str
            path to disparity image
        multiplier : float
            multiplier to convert saved intensities to disparities
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W disparity image
    '''

    # Load image and resize
    disparity = Image.open(path).convert('I')

    # Convert unsigned int16 to disparity values
    disparity = np.asarray(disparity, np.uint16)
    disparity = disparity / multiplier

    if disparity.ndim == 2:
        disparity = np.expand_dims(disparity, axis=-1)

    disparity = np.transpose(disparity, (2, 0, 1)).astype(np.float32)

    return torch.from_numpy(disparity)

def save_img(t, f):
    assert isinstance(t, (torch.Tensor, np.ndarray))
    if isinstance(t, torch.Tensor):
        if t.requires_grad:
            t = t.detach().cpu().numpy().squeeze()
        else:
            t = t.cpu().numpy().squeeze()
    else:
        t = t.squeeze()

    # normalize to [0, 255]
    # t = (t - t.min()) / (t.max() - t.min()) * 255.0
    t = t * 255.0
    # convert to HWC format if 3 channels
    if len(t.shape) == 3:
        t = t.transpose((1, 2, 0))
    
    Image.fromarray(t.astype(np.uint8)).save(f)

def save_perturb(t, f):
    assert isinstance(t, torch.Tensor), "only tensor is supported type for perturbation"
    # t = torch.sigmoid(t)
    # t = (torch.tanh(t)+1)/2
    # t = MinMaxNormalizeSTE.apply(t)
    # t = torch.clip(t, 0, 1)
    t = t.detach().cpu().numpy().squeeze()

    if t.shape[0] == 4:
        t = t[:3]  # discard alpha channel for saving

    if len(t.shape) == 3:
        t = t.transpose((1, 2, 0))

    t = t * 255.0
    # print(t.min(), t.max())
    # t = t.astype(np.uint8)
    # print(t.min(), t.max())

    # t = (t+1)/2 * 255.0
    # t = (t/0.02 +1)/2 * 255.0
    # t = (t - t.min()) / (t.max() - t.min()) * 255.0

    Image.fromarray(t.astype(np.uint8)).save(f)

def save_depth(disp, f, max_disp=None, cm=None, invert=False):

    disp = disp.detach().cpu().numpy().squeeze()

    if max_disp is not None:
        # filter all the invalid predictions
        valid_pred_mask = np.logical_and(disp>0, disp<max_disp)
    else:
        valid_pred_mask = disp>0
        max_disp = disp.max()
    disp = disp / max_disp * 255.0
    disp[~valid_pred_mask] = 0
    disp = disp.astype(np.uint8)

    # apply colormap
    if cm is not None:
        disp = cv2.applyColorMap(disp, cm)
    disp = (255 - disp) if invert else disp

    Image.fromarray(disp).save(f)

def lp_projection(v, output_norm, p='inf'):
    '''
    Project v onto lp ball centered at 0, radius output_norm

    Arg(s):
        v : torch.Tensor[float32]
            tensor
        output_norm : float
            radius of lp ball
        p : str
            specified lp ball
    Returns:
        torch.Tensor[float32] : tensor projected onto lp ball
    '''

    if p == 'inf':
        v = torch.sign(v) * torch.min(torch.abs(v), torch.tensor([output_norm], device=v.device))
    else:
        raise ValueError("Currently only supports p='inf'")

    return v

def relaxed_binary(x, tau=0.5):
    """
    Apply a temperature-controlled sigmoid to push values toward 0 or 1
    without killing gradients (relaxed binarization).
    
    Args:
        x (Tensor): Input mask values, assumed in [0, 1]
        tau (float): Temperature, lower = sharper push toward binary
    
    Returns:
        Tensor: Values in (0, 1), pushed toward binary but differentiable
    """
    return torch.sigmoid((x - 0.5) / tau)

def soft_binarize(x):
    """
    Pushes values in (0, 1) toward 0 or 1 by penalizing mid-range values.
    Input:
        x: Tensor with values in [0, 1]
    Output:
        loss: Tensor of same shape with penalty applied
    """
    return 4 * x * (1 - x)

