# -*- encoding: utf-8 -*-
"""
Template configuration for `opt_video_case.py` (temporal / physical video attack).

HOW TO USE
----------
1. Copy this file to a new name, e.g. `src/config/temporal/<model>_scene<NN>_frame<NNN>_id<N>.py`
   (see aanet.py / psmnet.py / raftstereo.py / sttr.py / deeppruner.py for worked examples).
2. Fill in every <PLACEHOLDER> below.
3. Run:
       python src/opt_video_case.py --cfg src/config/temporal/<your_config>.py

This file is loaded via `Config.fromfile()` (src/config/__init__.py): it is imported as a
plain python module and every top-level, non-dunder name becomes an attribute access on the
resulting `Config` object — e.g. the `exp = dict(gpu=0, ...)` block below becomes `cfg.exp.gpu`
inside opt_video_case.py. All six top-level dicts here (exp, dataset, model, optimizer, patch,
deploy) are read by that script; missing a required key raises an AttributeError at startup
before any model/scene loading happens.
"""

# ---------------------------------------------------------------------------
# exp: run bookkeeping — naming, device, logging, checkpoint cadence
# ---------------------------------------------------------------------------
exp = dict(
    gpu=0,                      # CUDA device index -> torch.device(f'cuda:{gpu}'); falls back to CPU if no CUDA
    name='<model>_vkitti2_scene<NN>_<variation>_frame<NNN>_id<N>',
                                 # experiment name; results are written to {save_dir}/{name}/
    round=200,                  # number of optimization rounds (one Adam step per round, gradients
                                 # accumulated over every frame in the deployed/visible window)
    n_checkpoint=10,            # every N rounds: save per-frame images/depth/masks + append results.txt
    save_dir='/home/yxing/projects/stereo_PhysicalAttack/results/temporal/distance',
                                 # base output directory, shared across experiments (each writes to its own {name}/ subfolder)
    log_level='debug',          # python logging level: debug | info | warning | error | critical
)

# ---------------------------------------------------------------------------
# dataset: which Virtual KITTI 2 scene/variation to attack
# ---------------------------------------------------------------------------
dataset = dict(
    name='vkitti2',
    root='/mnt/data/data_yxing/Virtual_KITTI2',   # dataset root, passed straight to VirtualKITTI2Loader
    scene='<NN>',                # one of: '01', '02', '06', '18', '20'
    variation='clone',           # one of: 'clone', '15-deg-left', '15-deg-right', '30-deg-left',
                                  #         '30-deg-right', 'fog', 'morning', 'overcast', 'rain', 'sunset'
)

# ---------------------------------------------------------------------------
# model: stereo network under attack + loss term weights
# ---------------------------------------------------------------------------
model = dict(
    name='<model>',               # one of: psmnet, deeppruner, aanet, raftstereo, sttr (see StereoModel)
                                   # NOTE: psmnet can currently only run on cuda:0 — set exp.gpu=0 if using it.
    ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/_checkpoints/<checkpoint_file>',
    loss=dict(
        unit_norm=True,           # NOTE: currently unused by opt_video_case.py. The equivalent --unit_norm
                                   # CLI flag IS read by the non-temporal opt_case.py/opt_dataset.py scripts,
                                   # but here the L-inf projection radius is hardcoded to 0.5 at the bottom of
                                   # the round loop (`lp_projection(board_manager._unit.data, 0.5, 'inf')`).
                                   # Kept for config-shape parity with those scripts; has no effect here.
        alpha=0.1,                 # weight on entropy_loss(opt_board) — pushes patch pixels toward 0/1 extremes
        beta=0.01,                  # weight on the GT-scale-normalized regional MSE — this is the actual attack term
        gamma=10,                   # weight on total_variation_loss(opt_board) — encourages smooth/printable regions
        delta=0.01,                  # reserved for an extra consistency/frequency term; currently not summed into
                                   # the loss (see the commented-out line where the round loss is assembled)
    )
)

# ---------------------------------------------------------------------------
# optimizer: Adam over the single optimizable texture unit (DPABoard._unit)
# ---------------------------------------------------------------------------
optimizer = dict(
    lr=0.01,     # REQUIRED. opt_video_case.py reads cfg.optimizer.lr directly when constructing the Adam
                 # optimizer — omitting this block raises AttributeError before anything else runs.
                 # 0.01 matches the default used by the non-temporal opt_*.py scripts; deeppruner.py uses
                 # 0.1 instead (tuned per-model), so treat this as a per-experiment knob, not a fixed constant.
)

# ---------------------------------------------------------------------------
# patch: geometry of the tiled adversarial texture
# ---------------------------------------------------------------------------
patch = dict(
    mode='given_size',    # 'given_size': use `size` (h, w) directly as the board's pixel dimensions.
                           # 'fit_size':   treat `size` as a *target* size, uniformly scaled down (aspect
                           #               preserved) to fit inside the largest axis-aligned rectangle found
                           #               within `deploy.frame_mask_left`.
    size=(128, 181),       # board size in pixels, (height, width)
    colored=False,         # False: optimize a single-channel unit, replicated to RGB when rendered/tiled.
                           # True:  optimize a 3-channel (RGB) unit directly.

    # Exactly one of the next two should be active — if both are set, unit_size wins (checked first).
    unit_size=(32, 36),    # (height, width) of the single optimizable tile; the tile count is then derived
                           # as n_ytiles, n_xtiles = board_size // unit_size.
    # yx_tiles=(4, 5),     # alternative: directly specify (n_ytiles, n_xtiles) and let unit_size be derived
                           # as board_size // yx_tiles. To use this instead, comment out unit_size above and
                           # uncomment this line.
)

# ---------------------------------------------------------------------------
# deploy: where in the scene the patch surface lives
# ---------------------------------------------------------------------------
deploy = dict(
    start_frame_idx=0,    # reference frame index (0-based). frame_mask_left is back-projected to 3D and the
                           # deployment plane is fit at this frame; optical-flow tracking then runs both
                           # forward and backward in time from here to cover the rest of the clip.
    frame_mask_left='/home/yxing/projects/stereo_PhysicalAttack/assets/masks/<scene>_frame<NNN>_som_id<N>_mask.png',
                           # single-channel (grayscale) mask, same resolution as the left camera frame,
                           # marking the target surface at start_frame_idx. opt_video_case.py asserts
                           # `len(mask.shape) == 2` — a 3-channel/RGBA mask file will fail that check.
                           # Generate with src/tools/get_som.py (SoM segmentation) or by hand.
)
