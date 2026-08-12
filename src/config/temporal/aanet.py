exp = dict(
    gpu=1,
    name='aanet_vkitti2_scene18_clone_frame296_id5_ratioLoss', # model_dataset_scene_variation_patch
    round=200,
    n_checkpoint=10,
    save_dir='/home/yxing/projects/stereo_PhysicalAttack/results/temporal/distance',
    log_level='debug',
)

dataset = dict(
    name='vkitti2',
    root='/mnt/data/data_yxing/Virtual_KITTI2',
    scene='18',
    variation='clone',
    # normalize=True, # normalize to [0,1] range
)

model = dict(
    name='aanet',
    ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/_checkpoints/aanet_kitti15-fb2a0d23.pth',
    loss = dict(
        unit_norm=True,
        alpha=0.1,
        beta=1.0,
        gamma=10,
        delta=0.01,
    )
)

optimizer = dict(
    lr=0.01
)



# # FOR VERIFICATION ONLY
# patch = dict(
#     file='/home/yxing/projects/stereo_PhysicalAttack/assets/patches/aanet.jpg',
#     mode='maximum_size', # 'given_size' or 'maximum_size'
#     given_width=40, # will not be used if mode is 'maximum_size'
#     given_height=40, # will not be used if mode is 'maximum_size'
# )

patch = dict(
    mode='given_size', # 'given_size' or 'fit_size'
    # size=(32, 36), # minimal to place a single unit
    # size=(256, 362), # x0.4
    # size=(256, 362), # x0.8
    size=(128, 181), # user given patch (height, width) in pixel
    # size=(256, 362), # x1.2
    # size=(256, 362), # x1.4
    # size=(256, 362), # x1.8
    # size=(256, 362), # x2.0
    colored=False,
    # yx_tiles=(4,5),
    unit_size=(32,36) # (height, width)
)

deploy = dict(
    start_frame_idx=296, # frame index starts from 0
    frame_mask_left='/home/yxing/projects/stereo_PhysicalAttack/assets/masks/scene18_frame296_som_id5_mask.jpg.png',
)