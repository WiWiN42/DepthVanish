# -*- encoding: utf-8 -*-
"""
@File    :   opt_video_case.py
@Time    :   2026/03/31 09:41:37
@Author  :   yxing
"""

import os
import sys
import math
import random
import shutil
import logging
import argparse
from pathlib import Path

import torch
import cv2
import numpy as np

from config import Config

from model.stereo_model import StereoModel
from utils.assemble import DPABoard, determine_board_size, find_maximum_rectangle
from utils.dataset import StereoDataset, VirtualKITTI2Loader
from utils.deploy import DigitalDeploy, StereoPatchDeployer
from utils.metric import cal_metric
from utils.loss import regional_mean_square_error, entropy_loss, total_variation_loss, disparity_ratio_loss, lagging_frame_penalty
from utils.tool import save_img, save_depth, save_perturb, load_image, lp_projection, round_numbers, fit_disparity_plane



parser = argparse.ArgumentParser()

parser.add_argument("--cfg", type=str, default="./config/temporal/template.py", help="the configuration to run the experiment")

args = parser.parse_args()

# Load configuration
cfg = Config.fromfile(args.cfg)

# setup device
if torch.cuda.is_available():
    device = torch.device(f'cuda:{cfg.exp.gpu}')
else:
    device = torch.device('cpu')

# setup saving root
save_dir = os.path.join(os.getcwd(), cfg.exp.save_dir, cfg.exp.name)
Path(save_dir).mkdir(parents=True, exist_ok=True)
# save training source code
# shutil.copy(__file__, save_dir)
# save configuration source
shutil.copy(args.cfg, save_dir)

# setup logger
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DPA-temporal")
logger.setLevel(getattr(logging, cfg.exp.log_level.upper()))
fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w') # erase previous logs
fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

logger.info("=" * 60)
logger.info(f"Configuration Setup")
logger.info(f"  Device: {device}")
logger.info(f"  Experiment: {cfg.exp.name}")
logger.info(f"  Save Directory: {save_dir}")
logger.info("=" * 60)

# ===== Reproducibility: fix the global random seed before any sampling =====
# The pipeline's randomness comes from RANSAC plane fitting, optical-flow tracking
# point sampling (both np.random) and any torch/cv2 RNG use. Seeding all of them
# makes a full run reproducible from round 1 (same patch, same round curve).
# seed = getattr(cfg.exp, 'seed', 0)
# random.seed(seed)
# np.random.seed(seed)
# cv2.setRNGSeed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# logger.info(f"Global random seed set to {seed} (deterministic mode, cudnn.benchmark disabled)")

# === Initialize Stereo Estimator ===
logger.info(f"Loading stereo model: {cfg.model.name}")
logger.debug(f"  Checkpoint: {cfg.model.ckpt}")
model = StereoModel(method=cfg.model.name, device=device)
model.restore_model(cfg.model.ckpt)
model.eval()
logger.info("✓ Stereo model loaded successfully")

# === Initialize Scene Loader ===
logger.info(f"Loading scene: {cfg.dataset.scene} (variation: {cfg.dataset.variation})")
scene_loader = VirtualKITTI2Loader(
    root_dir=cfg.dataset.root,
    scene=cfg.dataset.scene,
    variation=cfg.dataset.variation
)
logger.info(f"✓ Scene loader initialized with {len(scene_loader)} frames")

# === Prepare the target surface mask ===
logger.info(f"Loading surface mask from: {cfg.deploy.frame_mask_left}")
assert os.path.exists(cfg.deploy.frame_mask_left), f"Surface mask not found at: {cfg.deploy.frame_mask_left}"
surface_mask_left = cv2.imread(cfg.deploy.frame_mask_left, cv2.IMREAD_UNCHANGED) # ndarray, (h, w), uint8
surface_mask_left = (surface_mask_left/255.0).astype(np.float32) # normalize
assert len(surface_mask_left.shape) == 2, "The mask is supposed be a grayscale image."
logger.info(f"✓ Surface mask loaded with shape: {surface_mask_left.shape}")

# === Prepare the patch image ===
logger.info(f"Computing board dimensions")
# NOTE: below board size determination implicitly assumes the deployment surface faces towards the camera
if cfg.patch.mode == 'fit_size': # resize the patch to fit the largest rectangle within the masked region if the mode is 'fit_size'
    board_height, board_width = determine_board_size(surface_mask_left, cfg.patch.size, logger=logger)
elif cfg.patch.mode == 'given_size': # directly use the user given patch size if the mode is 'given_size'
    board_height, board_width = cfg.patch.size
    # Warn if the given patch size exceeds the mask region — the patch will overhang
    # beyond the physical surface, causing deploy_mask to include unrealistic pixels
    # and optical flow tracking to extrapolate beyond tracked feature points.
    (_, _, max_w, max_h), _ = find_maximum_rectangle(surface_mask_left)
    if board_height > max_h or board_width > max_w:
        logger.warning(
            f"Given patch size ({board_height}, {board_width}) exceeds the largest inscribed "
            f"rectangle in the mask ({max_h}, {max_w}). The patch will extend beyond the "
            f"target surface, which may cause inaccurate tracking and unrealistic deploy masks."
        )
elif cfg.patch.mode == 'freeform':
    # TODO the patch could also be deployed into the scene on a freeform billboard which can be hold anywhere
    pass
else:
    raise ValueError(f"Invalid patch mode: {cfg.patch.mode}. Supported modes are 'fit_size' and 'given_size'.")

# the unit size is either given by user or determined by dividing the board into tiles
if 'unit_size' in cfg.patch and cfg.patch.unit_size is not None:
    unit_height, unit_width = cfg.patch.unit_size
    n_ytiles = board_height // unit_height
    n_xtiles = board_width // unit_width
elif 'yx_tiles' in cfg.patch and cfg.patch.yx_tiles is not None:
    n_ytiles, n_xtiles = cfg.patch.yx_tiles
    unit_height = board_height // n_ytiles
    unit_width = board_width // n_xtiles
else:
    raise ValueError("Either 'unit_size' or 'yx_tiles' must be specified in the patch configuration.")

board_manager = DPABoard( # Depth Patch Attack Board
    unit_size=(unit_height, unit_width),
    board_size=(board_height, board_width),
    colored=cfg.patch.colored,
    device=device
)

logger.info(f"✓ Board configured as:")
logger.info("-" * 60)
logger.info(f"  Unit size: {(unit_height, unit_width)}, Board size: {(board_height, board_width)}")
logger.info(f"  Tiles: {n_ytiles}x{n_xtiles}, Colored: {cfg.patch.colored}")
logger.info("-" * 60)

# === Initialize patch deployer ===
logger.info("Initializing stereo patch deployer...")
board_deployer = StereoPatchDeployer(sceneloader=scene_loader)

unit_optimizer = torch.optim.Adam([board_manager._unit], lr=cfg.optimizer.lr)

# torch.autograd.set_detect_anomaly(True)
logger.info("✓ Optimizer initialized")
logger.info(f"  Type: Adam, Learning rate: {cfg.optimizer.lr}")
logger.info(f"  Total rounds: {cfg.exp.round}, Checkpoint interval: {cfg.exp.n_checkpoint}")
logger.info(f"  Loss weights - entropy: {cfg.model.loss.alpha}, TV: {cfg.model.loss.gamma}, consistency: {cfg.model.loss.delta}")
logger.info(f"  Consistency loss - EMA decay: 0.9")

# === Get Patch Deployment Requirements ===
logger.info("Preparing deployment geometry (3D fitting, homographies, tracking)...")
deployment_info = board_deployer.prepare_deployment(
    config=cfg.deploy,
    patch_size=(board_height, board_width),  # tuple (h, w)
    surface_mask=surface_mask_left, # ndarray, (h, w), float32
)
logger.info("✓ Deployment preparation complete")
logger.info(f"  Visible frame range: [{deployment_info.start_idx}, {deployment_info.end_idx}]")
logger.info(f"  Visible / Total frames: {sum(deployment_info.all_visibility)} / {len(scene_loader)}")
logger.info(f"  Patch 3D corners (left cam):\n{deployment_info.patch_corners_3d_left}")
logger.info(f"  Patch 3D corners (right cam):\n{deployment_info.patch_corners_3d_right}")



best_result = dict(
    round=0,
    d1=0,
    d1_std=0,
    epe=0,
    epe_std=0,
)

# ===== Pre-compute the deploy frame range once (static) =====
# The depth filter and per-frame weights depend only on the scene geometry, not on the
# optimization state, so compute them before the round loop and reuse them every round.
max_distance = 15.0  # meters
deploy_idx_range = range(deployment_info.start_idx, deployment_info.end_idx + 1)
valid_start_idx = None
for idx in deploy_idx_range:
    frame_depth = scene_loader.get_frame(frame_idx=idx)['depth']
    surface_mask_idx = deployment_info.all_masks[idx]
    surface_depth = frame_depth[surface_mask_idx > 0.5]
    valid_surface_depth = surface_depth[~np.isnan(surface_depth) & (surface_depth > 0)]
    if len(valid_surface_depth) > 0 and np.mean(valid_surface_depth) < max_distance:
        valid_start_idx = idx
        break
if valid_start_idx is not None:
    filtered_deploy_range = range(valid_start_idx, deploy_idx_range.stop)
else:
    raise ValueError("No valid deployment frames found within the specified depth threshold. Please check the surface mask and its depth.")
logger.info(f"Depth-filtered deploy frames: {len(filtered_deploy_range)} / {len(deploy_idx_range)} " f"(start idx: {valid_start_idx}, max surface depth threshold: {max_distance}m)")

for rnd in range(1, cfg.exp.round+1):
    logger.info(f"{'='*60}")
    logger.info(f"Round {rnd:03d}/{cfg.exp.round:03d}")
    logger.info(f"{'='*60}")

    # Reassemble opt_board from updated _unit each round.
    opt_board = board_manager.assemble_opt_board(n_ytiles, n_xtiles)
    opt_board = board_manager.preprocess(opt_board) # repeat & normalize
    logger.debug(f"[Step 1] Assembled opt_board from _unit: shape={opt_board.shape}, range=[{opt_board.min():.4f}, {opt_board.max():.4f}]")

    logger.debug("[Step 2] Embedding patch into stereo frames...")
    results_left, results_right, tracked_masks, _deploy_idx_range = board_deployer.render_patch_stereo(
        deployment_info=deployment_info,
        patch_img=opt_board,
    )

    logger.debug("[Step 3] Estimating disparity for perturbed stereo frames...")
    d1_error, epe_error = [], []
    round_losses = []  # per-frame loss values, for a proper round-level mean at the end

    unit_optimizer.zero_grad()  # zero grad once before the frame loop
    is_checkpoint_round = (rnd % cfg.exp.n_checkpoint == 0)

    if is_checkpoint_round:
        # create subdirectories for saving results
        subdirs = ['board', 'depth', 'unit', 'image', 'mask']
        patch_dir, depth_dir, unit_dir, img_dir, mask_dir = [
            os.path.join(save_dir, f'round{rnd:03d}', subdir) for subdir in subdirs
        ]
        for subdir in [patch_dir, depth_dir, unit_dir, img_dir, mask_dir]:
            Path(subdir).mkdir(parents=True, exist_ok=True)

        # board_manager._unit and opt_board are constant for the whole round (only updated by
        # unit_optimizer.step() after this loop finishes), so save each exactly once per
        # checkpoint round here instead of once per frame inside the loop below.
        unit_path = os.path.join(unit_dir, f'{rnd:03d}_unit.jpg')
        save_perturb(board_manager._unit+0.5, unit_path)
        logger.debug(f"  ✓ Unit patch saved")

        board_path = os.path.join(patch_dir, f'{rnd:03d}_board.jpg')
        save_perturb(opt_board, board_path)
        logger.debug(f"  ✓ Assembled board saved")

    for i, (imgL, imgR, deploy_mask) in enumerate(zip(results_left, results_right, tracked_masks)):

        if i not in filtered_deploy_range and not is_checkpoint_round:
            # This frame contributes nothing this round: it's outside filtered_deploy_range
            # (no loss/backward) and this isn't a checkpoint round (nothing gets saved). The
            # stereo model forward pass is by far the most expensive thing in this loop --
            # running it on every non-deploy frame of the clip, every round, for a result
            # that's immediately discarded (see the `del ... perturb_disp` below) dominates
            # this script's wall-clock time for no benefit. Skip it entirely here.
            continue

        # render_patch_stereo() hands back a torch.Tensor only when the patch was actually
        # rendered for this frame (all_visibility[i] and depth available); otherwise it's a
        # raw numpy frame copy.
        # For logging purpose, we still process to get the depth when no patch is deployed.
        if not torch.is_tensor(imgL):
            imgL = torch.from_numpy(imgL).float()
        if not torch.is_tensor(imgR):
            imgR = torch.from_numpy(imgR).float()

        # add batch dimension then move to device
        imgL = imgL.unsqueeze(0).to(device)
        imgR = imgR.unsqueeze(0).to(device)

        with torch.set_grad_enabled(i in filtered_deploy_range):
            perturb_disp = model.forward(imgL, imgR).squeeze()

        if i in filtered_deploy_range:

            if not np.any(deploy_mask):
                # render_patch_torch can legitimately return an all-empty visibility mask for
                # one frame within an otherwise-visible range (patch clipped outside the image,
                # occluded, or a degenerate homography). Indexing perturb_disp/gt_disp_fitted by
                # an empty mask below would silently produce NaN via torch.mean() on zero
                # elements, poisoning the whole round's accumulated gradient on _unit once
                # loss.backward() runs -- skip this frame instead.
                logger.warning(f"[Frame {i:03d}] Empty deploy_mask despite being in filtered_deploy_range -- skipping loss for this frame")
            else:
                # original losses
                mse_loss = regional_mean_square_error(perturb_disp, mask=deploy_mask)
                ent_loss = entropy_loss(opt_board)
                tv_loss = total_variation_loss(opt_board)

                # NOTE frame-wise attack effect unsatisfied loss NOTE

                ## get ground-truth disparity first
                gt_disp = scene_loader.compute_disparity_from_depth(scene_loader.get_frame(frame_idx=i)['depth'], frame_idx=i)
                # Convert gt_disp to tensor if it's a numpy array
                if isinstance(gt_disp, np.ndarray):
                    gt_disp = torch.from_numpy(gt_disp).float()

                # fit the ground-truth disparity to the patch region
                surface_mask = deployment_info.all_masks[i]
                valid_region = (surface_mask > 0.5) & deploy_mask
                gt_disp_fitted = fit_disparity_plane(gt_disp, valid_region, deploy_mask)

                # Normalize MSE by squared mean GT disparity so that close and far
                # frames contribute equally (detached — no gradient through GT).
                with torch.no_grad():
                    gt_scale = gt_disp_fitted[deploy_mask > 0.5].mean() ** 2 + 1e-6
                mse_loss = mse_loss / gt_scale

                # mse_loss = disparity_ratio_loss(perturb_disp, gt_disp_fitted, target_ratio=0.5, mask=deploy_mask)

                # overall loss — uniform per-frame weight; the scale-MSE normalization
                # above (mse / gt_scale^2) already balances close vs far frames.
                loss = cfg.model.loss.beta * mse_loss + cfg.model.loss.alpha * ent_loss + cfg.model.loss.gamma * tv_loss
                # + cfg.model.loss.beta*consis_loss + cfg.model.loss.delta*frequency_loss(opt_mask)

                logger.debug(f"[Frame {i:03d}] Loss breakdown - MSE: {mse_loss:.6f}, Entropy: {ent_loss:.6f}, TV: {tv_loss:.6f}, Total: {loss:.6f}")

                logger.debug(f"[Frame {i:03d}] Accumulating gradient...")
                # board_grad = autograd.grad(loss, opt_board)[0]
                loss.backward()
                round_losses.append(loss.item())

                d1, epe = cal_metric(perturb_disp.cpu(), gt_disp_fitted, deploy_mask)
                d1_error.append(d1)
                epe_error.append(epe)

        if is_checkpoint_round:

            # save the frames with patch deployed
            logger.info(f"[Round {rnd:03d}] Saving frames{i:03d}...")
            left_path = os.path.join(img_dir, f'{i:03d}_left.jpg')
            save_img(imgL, left_path)
            right_path = os.path.join(img_dir, f'{i:03d}_right.jpg')
            save_img(imgR, right_path)
            logger.debug(f"  ✓ Stereo images saved")

            # save the corresponding region mask
            logger.info(f"[Round {rnd:03d}] Saving masks{i:03d}...")
            mask_path = os.path.join(mask_dir, f'{i:03d}_mask.jpg')
            save_img(deploy_mask, mask_path)
            logger.debug(f"  ✓ Region mask saved")

            # save the perturbed depth
            logger.info(f"[Round {rnd:03d}] Saving predictions for frame{i:03d}...")
            depth_path = os.path.join(depth_dir, f'{i:03d}_depth.jpg')
            save_depth(
                perturb_disp,
                depth_path,
                # max_disp=150,
                cm=cv2.COLORMAP_JET,
                invert=True
            )
            logger.debug(f"  ✓ Depth map saved")


            # save for statistical analysis (for all frames with patch deployed)
            if i in filtered_deploy_range:
                with open(os.path.join(save_dir, f'round{rnd:03d}', 'results.txt'), 'a') as f:

                    # get the predicted average depth over the full patch region
                    pred_patch_disp = perturb_disp[deploy_mask > 0].detach().cpu().numpy()
                    avg_pred_patch_depth = np.mean((725.0087 * 0.532725) / pred_patch_disp)

                    # ground-truth average depth from the fitted disparity plane
                    gt_patch_disp = gt_disp_fitted[deploy_mask > 0].detach().cpu().numpy()
                    avg_gt_patch_depth = np.mean((725.0087 * 0.532725) / gt_patch_disp)

                    f.write(f"{i} {avg_pred_patch_depth:.4f} {avg_gt_patch_depth:.4f}\n")

            logger.info(f"[Checkpoint {rnd:03d}] All files saved successfully")

        # Free per-frame GPU memory
        del imgL, imgR, perturb_disp

    # Step optimizer once after accumulating gradients from all frames
    unit_optimizer.step()
    board_manager._unit.data = lp_projection(board_manager._unit.data, 0.5, 'inf')

    # Free per-round tensors and reclaim GPU memory
    del results_left, results_right, tracked_masks, opt_board
    torch.cuda.empty_cache()

    logger.debug("[Step 4] Computing average results for this video clip...")
    avg_d1 = np.average(d1_error)
    std_d1 = np.std(d1_error)
    avg_epe = np.average(epe_error)
    std_epe = np.std(epe_error)
    # log the average results for this round
    logger.info("-" * 60)
    logger.info(f"{'Metric':<15} {'Mean':>12} {'Std Dev':>12}")
    logger.info("-" * 60)
    logger.info(f"{'D1-Error':<15} {avg_d1:>12.4f} {std_d1:>12.4f}")
    logger.info(f"{'EPE':<15} {avg_epe:>12.4f} {std_epe:>12.4f}")
    logger.info("-" * 60)

    # update the best results
    if avg_d1 > best_result['d1'] or avg_epe > best_result['epe']:
        best_result['round'] = rnd
        best_result['d1'] = avg_d1
        best_result['d1_std'] = std_d1
        best_result['epe'] = avg_epe
        best_result['epe_std'] = std_epe
        logger.info(f"→ New best result found at round {rnd:03d}")

    if round_losses:
        logger.info(f"Round {rnd:03d} completed - Mean Loss: {np.mean(round_losses):.6f} ({len(round_losses)} frames)\n")
    else:
        logger.info(f"Round {rnd:03d} completed - Mean Loss: N/A (no frames contributed this round)\n")

logger.info("\n" + "=" * 60)
logger.info("OPTIMIZATION COMPLETE")
logger.info("=" * 60)
logger.info(f"Best result achieved at Round {best_result['round']:03d}:")
logger.info("-" * 60)
logger.info(f"{'Metric':<15} {'Mean':>12} {'Std Dev':>12}")
logger.info("-" * 60)
logger.info(f"{'D1-Error':<15} {best_result['d1']:>12.4f} {best_result['d1_std']:>12.4f}")
logger.info(f"{'EPE':<15} {best_result['epe']:>12.4f} {best_result['epe_std']:>12.4f}")
logger.info("-" * 60)
logger.info(f"Results saved to: {save_dir}")
logger.info("=" * 60)