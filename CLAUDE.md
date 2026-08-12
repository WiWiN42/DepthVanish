# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project on **physical adversarial attacks against stereo depth estimation**. It designs adversarial textures/patches that, when printed on physical boards and placed in a scene, cause stereo matching models (PSMNet, AANet, RAFT-Stereo, etc.) to produce erroneous depth estimates. The attacks are "physical" because the optimized patterns are deployed onto real-world planar surfaces with proper 3D geometry, occlusion handling, and multi-view consistency.

The project supports two deployment paradigms:
- **Static/digital-only**: Overlay a patch at fixed pixel coordinates on a single stereo pair (simpler, but not physical-world realistic).
- **Temporal/physical**: Place a patch on a 3D planar surface with proper camera projection, then track it across video frames using optical flow — the rendered result respects occlusion and depth ordering.

## Project Structure

```
├── src/                        # Main source package
│   ├── opt_case.py             # Single stereo pair optimization
│   ├── opt_dataset.py          # Dataset-level optimization
│   ├── mask_opt.py             # Binary mask optimization
│   ├── grid_opt.py             # Per-cell grid optimization
│   ├── board_opt.py            # Legacy board optimization
│   ├── opt_video_case.py       # Temporal/physical video attack
│   ├── opt_video_case_ori.py   # Original video attack variant
│   ├── opt_case_revise.py      # Revised single-case optimization
│   ├── opt_dataset_revise.py   # Revised dataset optimization
│   ├── eval_case.py            # Single-case evaluation
│   ├── eval_dataset.py         # Dataset evaluation
│   ├── gen_lora.py             # SD LoRA fine-tuning for texture generation
│   ├── scripts/                # Shell scripts with example args, one per opt_*.py entry point
│   │   ├── run_opt_case.sh
│   │   ├── run_opt_dataset.sh
│   │   ├── run_mask_opt.sh
│   │   └── run_grid_opt.sh
│   ├── config/                 # Configuration system (Config class, temporal/ configs)
│   ├── utils/                  # Core utilities
│   │   ├── assemble.py         # DPABoard: tiling, gradient aggregation
│   │   ├── deploy.py           # DigitalDeploy + StereoPatchDeployer
│   │   ├── dataset.py          # StereoDataset + VirtualKITTI2Loader
│   │   ├── loss.py             # Loss functions (MSE, entropy, TV, ratio, etc.)
│   │   ├── metric.py           # D1-all error, EPE
│   │   ├── tool.py             # Image I/O, lp projection, binarization STE
│   │   └── scheduler.py        # Learning rate / parameter schedulers
│   ├── model/                  # Stereo model wrappers + pretrained implementations
│   │   ├── stereo_model.py     # Unified wrapper (PSMNet, DeepPruner, AANet, RAFT-Stereo, STTR)
│   │   ├── _checkpoints/       # Pre-trained weights (.pth / .tar)
│   │   └── */                  # Individual model source directories
│   └── tools/                  # External tool integrations (e.g., SoM segmentation)
├── data/                       # Dataset file lists (text files of image paths)
├── results/                    # Experiment outputs (logs, checkpoints, metrics)
├── notebooks/                  # Jupyter notebooks for experiments and PoCs
├── assets/                     # Image assets organized by purpose
│   ├── textures/               # Texture seeds and base patterns
│   ├── figures/                # Paper figures and motivation examples
│   ├── patches/                # Optimized adversarial patch images
│   ├── masks/                  # Surface masks for temporal attack deployment
│   └── physical/               # Physical-world deployment photos
├── submission/                 # Paper submissions (NeurIPS, TPAMI)
└── guided_diffusion/           # Diffusion model utilities (DDNM-based)
```

## Key Entry Points

### Optimization (Adversarial Patch Generation)

Run scripts from the project root:

```bash
# Using shell scripts (run from project root):
bash src/scripts/run_opt_case.sh

# Or directly:
python src/opt_case.py \
  --exp test/aanet_test \
  --round 500 --n_checkpoint 10 \
  --model aanet --ckpt <path_to_ckpt> \
  --dataset KITTI \
  --physical_height 0.891 --physical_width 1.26 --physical_depth 5 \
  --img_left <left.png> --img_right <right.png> --stereo_calib <calib.txt>
```

| Script | Purpose |
|---|---|
| `src/opt_case.py` | Optimize a tiled-texture patch against a single stereo pair. Core single-image attack. |
| `src/opt_dataset.py` | Optimize against a full dataset (multiple stereo pairs). Iterates over data files. |
| `src/mask_opt.py` | Optimize a binary mask pattern (not full texture). Uses sigmoid annealing toward binary. |
| `src/grid_opt.py` | Optimize each grid cell independently (individual units per grid position, not tiled). |
| `src/board_opt.py` | Legacy board optimization — uses pre-computed RAFT-Stereo externally. |
| `src/opt_video_case.py` | Temporal attack: optimize a patch across video frames with 3D surface deployment. |
| `src/opt_case_revise.py` | Revised single-case optimization variant. |

### Evaluation

- `src/eval_case.py` — Evaluate a pre-optimized patch on a single stereo pair; reports D1-all error and EPE.
- `src/eval_dataset.py` — Evaluate across a dataset.

### Generation

- `src/gen_lora.py` — Fine-tune Stable Diffusion with LoRA for texture generation (uses `diffusers`/`peft`).

## Architecture

### Attack Pipeline (single-image: `opt_case.py`)

1. **Calibration**: `DigitalDeploy` loads stereo calibration, computes the pixel position/size of a patch given physical dimensions (meters) and depth.
2. **Tiling**: `DPABoard` takes a small optimizable "unit" and tiles it into a larger board that fits the computed pixel region.
3. **Embedding**: `DigitalDeploy.deploy()` overlays the board onto the left/right stereo images.
4. **Forward**: Patched stereo pair → stereo model → perturbed disparity.
5. **Loss**: Regional MSE between predicted and ground-truth disparity (maximized to fool the model), plus regularization (entropy, TV).
6. **Backprop**: Gradients flow through the model back to the unit texture. Gradients across tiled regions are aggregated.

### Attack Pipeline (temporal: `opt_video_case.py`)

1. **Surface fitting**: `StereoPatchDeployer.prepare_deployment()` back-projects a user-selected surface mask to 3D, fits a plane via RANSAC, places patch corners on the plane, computes patch-to-frame homographies, and tracks the surface across all frames using optical flow.
2. **Differentiable rendering**: `StereoPatchDeployer.render_patch_stereo()` renders the patch into every frame using `render_patch_torch()` — a fully differentiable renderer (`grid_sample`, sigmoid-based occlusion, alpha blending) that preserves gradients back to the patch tensor.
3. **Loss**: Multi-frame disparity-based losses (regional MSE, ratio loss, lagging-frame penalty) computed where the patch is visible.

### Core Classes

- **`DPABoard`** (`src/utils/assemble.py`): Manages the optimizable texture unit. Handles tiling the unit into a board, downsampling, binarization, and gradient aggregation across tiled regions. Supports colored (3-channel) or grayscale patterns.

- **`DigitalDeploy`** (`src/utils/deploy.py`): Simple stereo patch embedder. Uses camera calibration to project physical coordinates to pixel space, then pastes the patch at the computed location in both views.

- **`StereoPatchDeployer`** (`src/utils/deploy.py`): Full 3D-aware deployment for temporal/video attacks. Handles plane fitting, homography computation, optical-flow-based tracking, per-frame camera extrinsics, occlusion-aware rendering, and differentiable torch rendering. Uses VirtualKITTI2 dataset.

- **`StereoModel`** (`src/model/stereo_model.py`): Unified wrapper for stereo models (PSMNet, DeepPruner, AANet, RAFT-Stereo, STTR). Provides `forward()`, `compute_loss()`, `train()`, `eval()`, `restore_model()`.

- **`StereoDataset`** (`src/utils/dataset.py`): PyTorch Dataset for KITTI/DrivingStereo stereo pairs with optional ground truth disparity.

- **`VirtualKITTI2Loader`** (`src/utils/dataset.py`): Full loader for Virtual KITTI 2 with camera intrinsics/extrinsics, depth, optical flow (forward/backward), and stereo geometry.

## Key Design Decisions

- **Tiled unit optimization**: Rather than optimizing every pixel of a large patch, a small unit is tiled. This enforces a repeating pattern, which is both compact to print and naturally robust to minor misalignment.
- **Physical-to-pixel projection**: Patch placement is specified in physical units (meters) and projected to pixel coordinates using camera calibration, enabling real-world deployment.
- **Differentiable rendering for temporal attacks**: `render_patch_torch()` uses `grid_sample` and sigmoid-based occlusion so the entire pipeline (patch → rendered frames → stereo model → disparity → loss) is differentiable. This is critical for gradient-based patch optimization across video frames.
- **Straight-through estimator (STE)**: `SignMaskSTE` and `MinMaxNormalizeSTE` in `utils/tool.py` enable gradient flow through non-differentiable binarization/normalization operations.
- **Multiple stereo model backends**: The `StereoModel` wrapper supports PSMNet, DeepPruner, AANet, RAFT-Stereo, and STTR, enabling cross-model attack evaluation.

## Datasets

- **KITTI 2015**: Stereo driving dataset with calibration files. Images loaded via `StereoDataset`; calibration via `DigitalDeploy.load_calibrition()` with `P_rect_02`, `P_rect_03`, `R_rect_00`.
- **DrivingStereo**: Large-scale driving dataset with calibration. Uses `P_rect_101`, `P_rect_103`, `R_rect_101`.
- **Virtual KITTI 2**: Synthetic dataset with ground-truth depth, optical flow, and per-frame extrinsics. Used for temporal/physical attacks via `VirtualKITTI2Loader`.

## Common Paths

- Model checkpoints: `src/model/_checkpoints/`
- KITTI data: `/mnt/data/data_yxing/KITTI_stereo_2015/training/`
- DrivingStereo data: `/mnt/data/data_yxing/DrivingStereo/`
- Virtual KITTI 2: `/mnt/data/data_yxing/Virtual_KITTI2/`
- Results saved to: `results/<exp_name>/`
- Assets: `assets/` (organized into textures/, figures/, patches/, masks/, physical/)
- Data file lists: `data/`

## Code Style Notes

- The project uses `torch.Tensor` and `np.array` for construction.
- Image convention: PyTorch tensors are `(C, H, W)` with float32 values in `[0, 1]`. OpenCV images are `(H, W, C)` with uint8 values in `[0, 255]`. Conversion between the two is common.
- The `six` library is used for Python 2/3 compatibility in `config/`, though the main codebase requires Python 3.
- Logging uses Python's standard `logging` module with custom levels set via `--log_level`.
- Scripts use `argparse` for CLI and `Namespace` for programmatic configuration.
- Temporal/video experiments use Python config files loaded via `Config.fromfile()` (adapted from OpenMMLab pattern).
