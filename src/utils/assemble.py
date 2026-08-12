# -*- encoding: utf-8 -*-
"""
@File    :   assemble.py
@Time    :   2025/03/31 10:32:08
@Author  :   yxing
"""

"""
This file contains all the patch assemble related functions.
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Union

import torch
import numpy as np
import cv2
from torch.nn import functional as F
import torchvision.transforms.functional as TF

from .tool import relaxed_binary, lp_projection, soft_binarize, MinMaxNormalizeSTE, SignMaskSTE

class DPABoard():
    def __init__(self, unit_size=(10,20), board_size=(180,180), colored=False, device='cpu'):
        self.device = device
        self.colored = colored

        self.board_height, self.board_width = board_size
        self.unit_height, self.unit_width = unit_size
        assert (self.board_height >= self.unit_height) and (self.board_width >= self.unit_width), f"mask size {unit_size} is larger than board size {board_size} which is unsupported"

        self._unit = self.init_opt_unit()

    def tiled_horizontal_sines(self, freqs=[2, 4], tiles_per_freq=2):
        """
        Initialize a mask by tiling horizontal sine waves of given frequencies.
        
        Args:
            H (int): Total height of the mask.
            W (int): Width of the mask.
            freqs (list): Frequencies to cycle through (vertical spatial freq).
            tiles_per_freq (int): Number of times to vertically stack each wave.
            device (str): PyTorch device.

        Returns:
            mask (Tensor): Tensor of shape (1, H, W) with values in [0, 1].
        """
        total_tiles = len(freqs) * tiles_per_freq
        tile_height = self.unit_height // total_tiles
        mask = torch.zeros(self.unit_height, self.unit_width, device=self.device)

        current_y = 0
        for freq in freqs:
            for _ in range(tiles_per_freq):
                y = torch.linspace(0, 1, tile_height, device=self.device).unsqueeze(1)
                wave = torch.sin(2 * math.pi * freq * y)
                wave = 0.5 * (1 + wave / wave.abs().max())  # Normalize to [0, 1]
                wave_tile = wave.expand(tile_height, self.unit_width)
                mask[current_y:current_y + tile_height, :] = wave_tile
                current_y += tile_height

        # Pad if needed to fill H
        if current_y < self.unit_height:
            pad = torch.zeros(self.unit_height - current_y, self.unit_width, device=self.device)
            mask[current_y:] = pad

        return mask.requires_grad_(True)  # shape: (1, H, W)


    def multi_freq_horizontal_init(self, freqs=[1, 2, 3], weights=[0.5, 0.3, 0.2]):
        y = torch.linspace(0, 1, self.unit_height, dtype=torch.float32).unsqueeze(1)
        mask = sum(w * torch.sin(2 * torch.pi * f * y) for w, f in zip(weights, freqs))
        mask = 0.5 * (1 + mask / mask.abs().max())  # Normalize to [0, 1]
        mask = mask.expand(self.unit_height, self.unit_width).clone()

        return mask.to(self.device).requires_grad_(True)

    def patchy_horizontal_mask_init(self, patch_size=10, blur_sigma=2.0, sine_weight=0.4, freq=2):
        # ---- Step 1: Create coarse binary pattern ----
        coarse_H, coarse_W = self.unit_height // patch_size, self.unit_width // patch_size
        coarse_mask = torch.randint(0, 2, (1, 1, coarse_H, coarse_W)).float()

        # ---- Step 2: Upsample to full resolution ----
        upsampled = F.interpolate(coarse_mask, size=(self.unit_height, self.unit_width), mode='bilinear', align_corners=False)

        # ---- Step 3: Smooth the pattern ----
        smoothed = TF.gaussian_blur(upsampled[0], kernel_size=5, sigma=blur_sigma).unsqueeze(0)

        # ---- Step 4: Generate horizontal sine bias ----
        y = torch.linspace(0, 1, self.unit_height).unsqueeze(1)
        sine_wave = 0.5 * (1 + torch.sin(2 * torch.pi * freq * y))  # shape (H, 1)
        sine_mask = sine_wave.expand(self.unit_height, self.unit_width).unsqueeze(0)  # shape (1, H, W)

        # ---- Step 5: Blend them together ----
        hybrid = (1 - sine_weight) * smoothed + sine_weight * sine_mask
        hybrid = hybrid.clamp(0, 1)  # Ensure values stay in [0, 1]

        return hybrid.squeeze().to(self.device).requires_grad_(True)

    def init_opt_unit(self):
        # create mask basis
        if self.colored:
            _size = (3, self.unit_height, self.unit_width)
        else:
            _size = (1, self.unit_height, self.unit_width)
        _unit = torch.zeros(
            size=_size, 
            dtype=torch.float32,
            device=self.device
        )
        return _unit.requires_grad_(True)
    
    def mask_downsample(self, mask, op='interpolate', **kwargs):
        """Down-sample the mask for assembling the board."""
        valid_op = ['interpolate', 'conv', 'pool']

        assert len(mask.size()) == 2
        mask = mask.unsqueeze(0).unsqueeze(0)

        if op=='interpolate':
            # scale the amplified mask before assembling
            _mask = F.interpolate(
                mask, # spatial interpolate
                size=(self.unit_height, self.unit_width),
                mode='bilinear'
            )
        elif op == 'pool':
            if 'stride' not in kwargs.keys():
                s, k = 1, (self.unit_height*(self.amp_scale-1)+1, self.unit_width*(self.amp_scale-1)+1)
            else:
                s = kwargs['stride']
                assert s <= self.amp_scale, "the stride should not be larger than the mask scale factor"
                k = (self.unit_height*(self.amp_scale-s)+s, self.unit_width*(self.amp_scale-s)+s)
            _mask = F.avg_pool2d(mask, kernel_size=k, stride=s)
        else:
            raise ValueError(f"Unsupported mask down-sample operation {op}, please choose from {valid_op}")
        
        return _mask.squeeze(0)

    def board_downsample(self, board, op='interpolate', **kwargs):
        """Down-sample the board for assembling the stereo pairs."""
        valid_op = ['interpolate', 'pool']

        assert len(board.size()) == 2
        board = board.unsqueeze(0).unsqueeze(0)

        if op=='interpolate':
            # scale the amplified mask before assembling
            _board = F.interpolate(
                board, # spatial interpolate
                size=(self.board_height, self.board_width),
                mode='bilinear'
            )
        elif op == 'pool':
            if 'stride' not in kwargs.keys():
                s, k = 1, (self.board_height*(self.amp_scale-1)+1, self.board_width*(self.amp_scale-1)+1)
            else:
                s = kwargs['stride']
                assert s <= self.amp_scale, "the stride should not be larger than the mask scale factor"
                k = (self.board_height*(self.amp_scale-s)+s, self.board_width*(self.amp_scale-s)+s)
            _board = F.avg_pool2d(board, kernel_size=k, stride=s)
        else:
            raise ValueError(f"Unsupported mask down-sample operation {op}, please choose from {valid_op}")
        
        return _board.squeeze(0)

    def get_opt_board(self, opt_mask):

        _board_base = torch.full(
            size=(3, self.board_height, self.board_width), 
            fill_value=255.0, dtype=torch.float32, device=self.device
        )

        _mask = self.mask_downsample(opt_mask, op='pool', stride=3)
        # first clamp the mask value
        _mask = self.soft_binarize_mask(_mask)
        # transform the mask to fit image form
        _mask = (_mask*255.0).repeat(3,1,1)

        n_repeat_x, x_res = divmod(self.board_width, self.unit_width)
        n_repeat_y, y_res = divmod(self.board_height, self.unit_height)

        # assemble the optimization mask into the board
        fill_x0, fill_y0 = x_res//2, y_res//2
        for j in range(0, n_repeat_y):
            for i in range(0, n_repeat_x):
                _board_base[:,fill_y0+self.unit_height*j:fill_y0+self.unit_height*(j+1),fill_x0+self.unit_width*i:fill_x0+self.unit_width*(i+1)] = _mask

        return _board_base
    
    def assemble_opt_board(self, y_tile, x_tile):
        """Tile the optimizable texture unit into a board that fit into predefined ratio."""

        x_res = self.board_width % self.unit_width
        y_res = self.board_height % self.unit_height

        num_repeat = (1, y_tile, x_tile) if self.colored else (3, y_tile, x_tile)
        _board = self._unit.repeat(num_repeat)

        # pad the board evenly to fit the target board size if the unit size does not divide the board size perfectly
        pad_x = (self.board_width - _board.shape[2]) // 2
        pad_y = (self.board_height - _board.shape[1]) // 2

        _board = F.pad(_board, (pad_x, x_res - pad_x, pad_y, y_res - pad_y), value=0)

        return _board
    

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure the patch has three channels and normalize to [0, 1] range.
        
        NOTE This function repeats the patch to RGB if it is grayscale, then
        normalizes from _unit range [-0.5, 0.5] to [0, 1].
        
        Args:
            tensor: Input tensor of shape (C, H, W) where C is 1 or 3 (grayscale or RGB)
        
        Returns:
            Tensor of shape (3, H, W) in [0, 1] range
        """
        device = tensor.device
        C, H, W = tensor.shape

        if C == 1:
            # Repeat grayscale to RGB
            tensor = tensor.repeat(3, 1, 1)
        
        # Normalize RGB from _unit range [-0.5, 0.5] to [0, 1] for rendering and loss functions
        tensor = tensor + 0.5
        
        return tensor

    def assemble_grid_board(self):
        """Tile the optimizable mask unit into a board that fit into predefined ratio."""
        num_repeat = (1, 1, 1) if self.colored else (3, 1, 1)

        _board = torch.ones((3, self.board_height, self.board_width), requires_grad=False) # only the texture element requires gradients
        for i in range(0,5): # repeat horizontally
            for j in range(0,4): # repeat vertically
                start_y, end_y = j*(self.unit_height+10), j*(self.unit_height+10)+self.unit_height
                start_x, end_x = i*(self.unit_width+10), i*(self.unit_width+10)+self.unit_width
                _board[..., start_y:end_y, start_x:end_x] = self._unit.repeat(num_repeat)

        return _board

    def update_mask(self, mask_grad, anneal_para=None):
        """Update the mask with the aggregated gradient.

        Args:
            mask_grad (torch.Tensor): The gradient to be updated.
            norm (int): The size of the norm ball. Defaults to 0.0001.
        """
        # update the mask unit
        self._unit = self._unit - mask_grad
        # constraint the unit search space
        # self._unit = lp_projection(self._unit, 1, p='inf')
        if anneal_para is not None:
            assert isinstance(anneal_para, float), "anneal_para must be a float"
            self._unit = torch.sigmoid(anneal_para*self._unit)
        else:
            self._unit = torch.sigmoid(self._unit)

def aggregate_board_grad(grad, unit_size, colored=False, constraint=None):
    """Aggregate the gradients of the board.
    Args:
        grad (torch.Tensor): The gradient to be aggregated.
        unit_size (tuple): Size of the mask.
        mode (str): Aggregation mode. Defaults to 'mean'.
    Returns:
        torch.Tensor: The aggregated gradient.
    """
    # project the gradients into lp ball
    if constraint is not None:
        assert isinstance(constraint, float), "the gradient constraint must be specified as a number"
        grad = lp_projection(grad, constraint, p='inf')
    
    n_repeat_x, x_res = divmod(unit_size[1], unit_size[1])
    n_repeat_y, y_res = divmod(unit_size[0], unit_size[0])

    # aggregate the gradient over mask regions
    agg_grad = 0
    for i in range(n_repeat_x):
        for j in range(n_repeat_y):
            x_start = i * unit_size[1]
            y_start = j * unit_size[0]
            x_end = x_start + unit_size[1]
            y_end = y_start + unit_size[0]

            agg_grad += torch.squeeze(grad[:, y_start:y_end, x_start:x_end])
            assert agg_grad.shape == (3, unit_size[0], unit_size[1]), f"agg_grad shape mismatch: {agg_grad.shape} != {(3, unit_size[0], unit_size[1])}"

    agg_grad = agg_grad/(n_repeat_x*n_repeat_y)
    if colored:
        return agg_grad
    else:
        return torch.mean(agg_grad, axis=0)

def check_stereo_consistency(results_left, results_right, frame_idx):
    """Check if patch appears at correct depth in stereo."""
    left_frame = results_left[frame_idx]
    right_frame = results_right[frame_idx]
    
    # Simple check: patch should be displaced horizontally
    # according to its depth (disparity = baseline * focal_length / depth)
    
    # Convert to grayscale for correlation
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute stereo disparity (simplified)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=11
    )
    disparity = stereo.compute(left_gray, right_gray)
    
    # Check if patch regions have plausible disparity
    return disparity

def count_regions(region_mask: np.ndarray) -> int:
    """Count connected regions in a 2D mask.

    Args:
        region_mask: 2D numpy array (H, W) of numeric or boolean values.

    Returns:
        Number of connected foreground regions.
    """
    if not isinstance(region_mask, np.ndarray):
        raise TypeError(f"region_mask must be a numpy array, got {type(region_mask)}")
    if region_mask.ndim != 2:
        raise ValueError(f"region_mask must be 2D (H, W), got shape {region_mask.shape}")
    if region_mask.size == 0:
        return 0
    if not (np.issubdtype(region_mask.dtype, np.number) or region_mask.dtype == np.bool_):
        raise TypeError(f"region_mask must be numeric or boolean, got {region_mask.dtype}")

    binary_mask = (region_mask > 0).astype(np.uint8)
    num_labels, _labels = cv2.connectedComponents(binary_mask)
    num_regions = num_labels - 1

    return num_regions

def find_maximum_rectangle(mask: np.ndarray) -> Tuple[Tuple[int, int, int, int], int]:
    """
    Find the maximum area axis-aligned rectangle in a binary mask.
    
    Uses the maximal rectangle in histogram algorithm with dynamic programming.
    Ensures the resulting rectangle has NO rotation - strictly axis-aligned.
    
    Args:
        mask: Binary mask where True/non-zero indicates target region
        
    Returns:
        Tuple of ((x, y, width, height), area) representing the maximum rectangle
        - (x, y): top-left corner coordinate
        - width, height: dimensions of the rectangle
        - area: the area of the rectangle
    """
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    h, w = binary_mask.shape
    
    if h == 0 or w == 0:
        return (0, 0, 1, 1), 0
    
    # Build height matrix using dynamic programming
    # heights[i][j] = consecutive 1s above (i,j) including (i,j)
    heights = np.zeros((h, w), dtype=np.int32)
    
    for i in range(h):
        for j in range(w):
            if binary_mask[i, j] > 0:
                heights[i, j] = heights[i-1, j] + 1 if i > 0 else 1
            else:
                heights[i, j] = 0
    
    max_area = 0
    best_rect = (0, 0, 1, 1)
    
    # For each row, find maximum axis-aligned rectangle in histogram
    for i in range(h):
        # Use stack-based approach (Largest Rectangle in Histogram algorithm)
        # This guarantees axis-aligned rectangles
        stack = []  # Stack of (index, height) pairs
        
        for j in range(w):
            h_val = heights[i, j]
            start = j
            
            # Pop taller bars and calculate areas
            while stack and stack[-1][1] > h_val:
                idx, height = stack.pop()
                area = height * (j - idx)
                
                if area > max_area:
                    max_area = area
                    # Rectangle position and dimensions:
                    # - x: idx (leftmost position of this histogram bar)
                    # - y: i - height + 1 (top of the rectangle)
                    # - width: j - idx (extends from idx to j)
                    # - height: height (height of the histogram bar)
                    best_rect = (idx, i - height + 1, j - idx, height)
                
                start = idx
            
            # Push current bar if it has height
            if h_val > 0:
                stack.append((start, h_val))
        
        # Process remaining bars in stack
        for idx, height in stack:
            area = height * (w - idx)
            
            if area > max_area:
                max_area = area
                best_rect = (idx, i - height + 1, w - idx, height)
    
    return best_rect, max_area

def determine_board_size(region_mask: np.ndarray, given_patch_size: Tuple, logger: logging.Logger) -> Tuple:
    """
    Determine a valid board size that fits within the largest axis-aligned rectangle of the masked region.

    If a target patch size is provided, it is returned when it fits; otherwise it is uniformly scaled down to preserve aspect ratio and fit within the maximum rectangle. If no target size is provided, the largest rectangle is used.

    NOTE The patch is supposed to be mounted on a physical board, thus they have different sizes during physical deployment. As we are conducting digital optimization, they share the same size during fabrication.

    Args:
        region_mask: Binary or continuous mask defining valid placement regions.
        given_patch_size: Optional target size as (height, width).
        logger: Logger for debug messages.

    Returns:
        Tuple of (height, width) for the selected board size.
    """
    num_regions = count_regions(region_mask)
    assert num_regions == 1, f"Expected exactly 1 masked region, got {num_regions}"

    (x, y, max_width, max_height), area = find_maximum_rectangle(region_mask)
    logger.debug(f"Maximum rectangle: position=({x}, {y}), size=({max_width}x{max_height}), area={area}")


    board_height, board_width = given_patch_size
    assert isinstance(board_width, int) and isinstance(board_height, int), \
        "Board size must be specified as integers when provided"
    
    if board_width <= max_width and board_height <= max_height:
        return (board_height, board_width)
    else:
        # Scale down while maintaining aspect ratio
        scale_w = max_width / board_width
        scale_h = max_height / board_height
        scale = min(scale_w, scale_h)
        
        height = int(board_height * scale)
        width = int(board_width * scale)
        
        logger.warning(f"Given size (h={board_height}, w={board_width}) exceeds "
                f"maximum available space (h={max_height}, w={max_width}). "
                f"Scaled to (h={height}, w={width}) maintaining aspect ratio.")

        return (height, width)