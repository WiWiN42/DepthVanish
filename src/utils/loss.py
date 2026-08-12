# -*- encoding: utf-8 -*-
"""
@File    :   loss.py
@Time    :   2025/04/01 11:04:42
@Author  :   yxing
"""

import torch
from torch.nn import functional as F


def regional_mean_square_error(pred_disp, tl_con=None, board_size=None, gt_disp=None, mask=None):
    """
    Computes the regional mean square error (MSE) for a given disparity map.

    NOTE Use mask if provided, top left (tl_con) coordinates and board size otherwise.

    Args:
        pred_disp (torch.Tensor): Predicted disparity map of shape (H, W) or (1, H, W).
        tl_con (tuple): Top-left corner coordinates (x, y) of the region.
        board_size (tuple): Size of the region (height, width).
        gt_disp (torch.Tensor, optional): Ground truth disparity map of shape (H, W) or (1, H, W).
        mask (torch.Tensor, optional): Mask tensor to apply to the region.
    Returns:
        torch.Tensor: Regional mean square error.
    """
    # use mask to identify region
    if mask is not None:
        assert mask.shape == pred_disp.shape, "mask and predicted disparity must have the same shape"
        region_pred_disp = pred_disp[mask]
        if gt_disp is not None:
            region_gt_disp = gt_disp[mask]
            rMSE = torch.mean(torch.square(region_pred_disp - region_gt_disp))
        else:
            rMSE = torch.mean(torch.square(region_pred_disp))
        return rMSE

    # use location and size to identify region
    if pred_disp.ndim != 2:
        pred_disp = pred_disp.squeeze()
    assert pred_disp.ndim == 2, "the predicted disparity has a invalid size"

    assert isinstance(tl_con, tuple), 'only tuple of coordinates is support to specify image top left corner'
    
    start_y, end_y = tl_con[1], tl_con[1]+board_size[0]
    start_x, end_x = tl_con[0], tl_con[0]+board_size[1]

    region_pred_disp = pred_disp[start_y:end_y, start_x:end_x]

    if gt_disp is not None:
        if gt_disp.ndim != 2:
            gt_disp = gt_disp.squeeze()
        assert gt_disp.ndim == 2, "the clean disparity has a invalid size"
        region_clean_disp = gt_disp[start_y:end_y, start_x:end_x]

    if gt_disp is not None:
        rMSE = torch.mean(torch.square(region_pred_disp-region_clean_disp))
    else:
        rMSE = torch.mean(torch.square(pred_disp[start_y:end_y, start_x:end_x]))
    return rMSE


def regional_smooth_l1_loss(pred_disp, tl_con=None, board_size=None):
    if pred_disp.ndim != 2:
        pred_disp = pred_disp.squeeze()
    assert pred_disp.ndim == 2, "the predicted disparity has a invalid size"
    if tl_con is None:
        rSLL = F.smooth_l1_loss(pred_disp, torch.zeros_like(pred_disp), reduction='mean')
    else:
        assert isinstance(tl_con, tuple), 'only tuple of coordinates is support to specify image top left corner'
        _pred = pred_disp[tl_con[1]:tl_con[1]+board_size[0],tl_con[0]:tl_con[0]+board_size[1]]
        rSLL = F.smooth_l1_loss(_pred, torch.zeros_like(_pred), reduction='mean')
        
    return rSLL

def non_printability_score():
    pass

def entropy_loss(_in: torch.Tensor, eps=1e-6):
    """
    Computes per-pixel binary entropy loss over the input of shape (C, H, W).
    Encourages values to be close to 0 or 1.
    
    Args:
        _in (torch.Tensor): Tensor of shape (C, H, W) with values in [0, 1].
        eps (float): Small value to avoid log(0).

    Returns:
        torch.Tensor: Scalar entropy loss.
    """
    if _in.dim() == 2:
        _in = _in.unsqueeze(0)
    assert _in.dim() == 3  # (1, H, W)

    _in = torch.clamp(_in, eps, 1 - eps)  # Ensure numerical stability
    loss = - (_in * torch.log(_in) + (1 - _in) * torch.log(1 - _in))
    return loss.mean()

def contrast_loss(_in: torch.Tensor):
    """
    Computes the negative variance of the input ensures that neighboring pixels aren't too similar if they're on opposite sides of a region boundary.
    
    Args:
        _in (torch.Tensor): Tensor of shape (C, H, W).
    
    Returns:
        torch.Tensor: Scalar contrast loss (lower means higher contrast).
    """
    if _in.dim() == 2:
        _in = _in.unsqueeze(0)
    assert _in.dim() == 3  # (1, H, W)

    mean = _in.mean()
    var = ((_in - mean) ** 2).mean()
    return -var  # Maximize variance -> minimize negative variance

def total_variation_loss(_in: torch.Tensor):
    """
    Computes total variation loss for a tensor of shape (C, H, W). Encourage adjacent pixels to have similar values unless there's a real boundary.

    Args:
        _in (torch.Tensor): Input tensor with values in [0, 1].

    Returns:
        torch.Tensor: Scalar TV loss.
    """
    if _in.dim() == 2:
        _in = _in.unsqueeze(0)
    assert _in.dim() == 3  # (1, H, W)

    dx = torch.abs(_in[:, :, 1:] - _in[:, :, :-1])  # horizontal diffs
    dy = torch.abs(_in[:, 1:, :] - _in[:, :-1, :])  # vertical diffs
    return (dx.mean() + dy.mean())

def disparity_ratio_loss(pred_disp: torch.Tensor, gt_disp: torch.Tensor, target_ratio: float, mask: torch.Tensor, eps: float = 1e-6):
    """
    Ratio-based disparity loss that is scale-invariant across frames.
    Penalizes deviation of (pred / gt) from target_ratio, uniformly regardless
    of the absolute disparity magnitude (i.e., surface distance).

    Loss = mean( (pred[mask] / gt[mask] - target_ratio)^2 )

    Args:
        pred_disp (torch.Tensor): Predicted disparity map, shape (H, W).
        gt_disp (torch.Tensor): Ground-truth disparity map, shape (H, W).
        target_ratio (float): Desired pred/gt ratio (e.g. 0.5 to halve disparity).
        mask (torch.Tensor): Boolean mask, shape (H, W), selects pixels to include.
        eps (float): Small value added to gt denominator to avoid division by zero.

    Returns:
        torch.Tensor: Scalar loss.
    """
    region_pred = pred_disp[mask]
    region_gt = gt_disp[mask].to(pred_disp.device)
    ratio = region_pred / (region_gt + eps)
    return torch.mean((ratio - target_ratio) ** 2)


def frequency_penalty_fft(_in: torch.Tensor, power=2.0, eps=1e-8):
    """
    Differentiable frequency penalty using 2D FFT.
    Penalizes high-frequency components in the input.
    
    Args:
        _in (Tensor): Tensor of shape (H, W) or (1, H, W)
        power (float): Exponent for frequency weighting (default=2.0)
        eps (float): Small constant to avoid divide-by-zero.
    Returns:
        Scalar tensor: frequency penalty loss
    """
    if _in.dim() == 2:
        _in = _in.unsqueeze(0)
    assert _in.dim() == 3  # (1, H, W)
    
    _, H, W = _in.shape

    # Compute 2D FFT and magnitude spectrum
    fft = torch.fft.fft2(_in)
    fft_mag = torch.abs(fft)

    # Shift zero freq to center
    fft_mag = torch.fft.fftshift(fft_mag)

    # Build frequency weight _in
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=1.0)).to(_in.device)
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=1.0)).to(_in.device)
    fy = fy.view(-1, 1)  # (H, 1)
    fx = fx.view(1, -1)  # (1, W)

    freq_magnitude = torch.sqrt(fx ** 2 + fy ** 2) + eps  # avoid div by zero
    freq_weights = freq_magnitude ** power  # (H, W)

    # Apply weights to FFT magnitude spectrum
    weighted_spectrum = fft_mag[0] ** 2 * freq_weights
    loss = weighted_spectrum.mean()
    
    return loss

def frequency_loss(_in, sigma=1.0, weight_horizontal=1.0, weight_vertical=4.0):
    """
    Compute frequency loss with:
    - Gaussian smoothing before FFT
    - Directional penalty to encourage horizontal structure (by penalizing vertical frequency)
    
    Args:
        _in: Tensor of shape (H, W) or (C, H, W)
        sigma: Standard deviation for Gaussian smoothing
        weight_horizontal: Weight for horizontal frequency penalty (lower = encourage horizontal features)
        weight_vertical: Weight for vertical frequency penalty
    """
    if _in.dim() == 2:
        _in = _in.unsqueeze(0)
    assert _in.dim() == 3  # (1, H, W)

    # -------------------------------
    # 1. Gaussian Smoothing (anti-alias)
    # -------------------------------
    def get_gaussian_kernel(size=5, sigma=1.0):
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel_1d = get_gaussian_kernel(kernel_size, sigma).to(_in.device)
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]  # outer product to get 2D Gaussian
    kernel_2d = kernel_2d[None, None, :, :]  # shape: (1, 1, k, k)

    mask_smooth = F.conv2d(_in.unsqueeze(0), kernel_2d, padding=kernel_size // 2)[0]

    # -------------------------------
    # 2. FFT and Directional Penalty
    # -------------------------------
    f = torch.fft.fft2(mask_smooth)
    fshift = torch.fft.fftshift(f)
    magnitude = torch.abs(fshift)

    H, W = magnitude.shape[-2:]
    y = torch.linspace(-1, 1, steps=H, device=_in.device)
    x = torch.linspace(-1, 1, steps=W, device=_in.device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Penalize vertical frequencies (to encourage horizontal features)
    freq_penalty = (weight_horizontal * xx**2 + weight_vertical * yy**2)

    loss = torch.mean(magnitude * freq_penalty)
    return loss


def target_anchored_disparity_loss(pred_disp: torch.Tensor, gt_disp: torch.Tensor, mask: torch.Tensor, target_ratio: float = 0.0, eps: float = 1e-6):
    """
    Anchors the mean predicted disparity in the patch region to a fixed fraction
    of the ground-truth disparity. Because every frame is pulled toward the same
    ratio, attack consistency across frames emerges implicitly.

    Loss = ( mean(pred[mask]) / (mean(gt[mask]) + eps) - target_ratio )^2

    Args:
        pred_disp (torch.Tensor): Predicted disparity map, shape (H, W).
        gt_disp (torch.Tensor): Ground-truth disparity map, shape (H, W).
        mask (torch.Tensor): Boolean or float mask, shape (H, W).
        target_ratio (float): Desired ratio of pred/gt mean disparity.
            0.0 means push disparity to zero (infinite depth illusion).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Scalar loss.
    """
    mean_pred = pred_disp[mask > 0.5].mean()
    mean_gt = gt_disp[mask > 0.5].to(pred_disp.device).mean()
    ratio = mean_pred / (mean_gt + eps)
    return (ratio - target_ratio) ** 2


def lagging_frame_penalty(pred_disp: torch.Tensor, mask: torch.Tensor, ema_value: float):
    """
    Asymmetric consistency loss that only penalizes frames where the attack is
    LESS effective than the running average (i.e., mean disparity > EMA).
    Frames that are more effective than average receive zero penalty.

    Loss = max(0, mean(pred[mask]) - ema)^2

    This cooperates with MSE (push disparity to zero):
    - MSE provides uniform pressure toward zero on all frames.
    - This loss adds EXTRA pressure on frames that are falling behind,
      without holding back frames that are ahead.
    - As average effectiveness improves round-over-round, the EMA drops,
      continuously raising the bar for lagging frames.

    Usage:
        Initialize ema = None before the frame loop.
        After computing this loss, update ema outside the gradient tape:
            with torch.no_grad():
                current = pred_disp[mask > 0.5].mean().item()
                ema = current if ema is None else decay * ema + (1 - decay) * current

    Args:
        pred_disp (torch.Tensor): Predicted disparity map, shape (H, W).
        mask (torch.Tensor): Boolean or float mask, shape (H, W).
        ema_value (float): The current EMA estimate (detached, no gradient).

    Returns:
        torch.Tensor: Scalar loss (zero if frame is at or below EMA).
    """
    mean_pred = pred_disp[mask > 0.5].mean()
    gap = mean_pred - ema_value
    return torch.clamp(gap, min=0.0) ** 2
