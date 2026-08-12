# -*- encoding: utf-8 -*-
"""
@File    :   metric.py
@Time    :   2025/05/09 11:20:51
@Author  :   yxing
"""

import numpy as np

def d1_error(src, tgt, k=3):
    '''
    D1 error reported for KITTI 2015.
    A pixel is considered to be correctly estimated if the disparity end-point error is < 3 px or < 5 %.

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : d1 error between source and target (percentage of pixels)
    '''

    E = np.abs(src - tgt)

    # original error
    n_err = np.count_nonzero(
        np.logical_and(
            (tgt > 0), 
            np.logical_and(
                E > 3, 
                (E/np.abs(tgt)) > 0.05
            )
        )
    )
    # add our requirement
    # n_err = np.count_nonzero(
    #     np.logical_and(
    #         (tgt > 0), 
    #         np.logical_and(
    #             np.logical_and(
    #                 E > 3, 
    #                 (E/np.abs(tgt)) > 0.05
    #             ),
    #             # np.abs(src) < np.min(tgt[tgt > 0])/k # the copilot modification for avoid 0 results
    #             np.abs(src) < np.min(tgt)/k # the original version of our attack target
    #         )
    #     )
    # )
    n_total = np.count_nonzero(tgt > 0)

    return n_err/n_total

def end_point_error(src, tgt):
    '''
    Computes end point error for scene flow datasets

    Calls mean absolute error, separate function for ease of naming

     Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error between source and target
    '''

    return mean_abs_err(src, tgt)

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error between source and target
    '''

    return np.mean(np.abs(src - tgt))

def cal_metric(pred, gt, mask=None):
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    pred = pred.detach().cpu().numpy().squeeze()
    gt = gt.detach().cpu().numpy().squeeze()
    
    assert pred.shape == gt.shape, 'the size of ground-truth and prediction results not match'

    return d1_error(pred, gt), end_point_error(pred, gt)