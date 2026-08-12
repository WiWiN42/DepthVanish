exp = dict(
    gpu=3,
    name='sttr_vkitti2_scene18_clone_frame296_id5_base', # model_dataset_scene_variation_patch
    round=200,
    n_checkpoint=1,
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
    name='sttr', 
    ckpt='/home/yxing/projects/stereo_PhysicalAttack/src/model/_checkpoints/kitti_finetuned_model.pth.tar', # psmnet
    loss = dict(
        unit_norm=True,
        alpha=0.1,
        beta=0.01,
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
    size=(128, 181), # user given patch (height, width) in pixel
    colored=False,
    # yx_tiles=(4,5),
    unit_size=(32,36) # (height, width)
)

deploy = dict(
    start_frame_idx=296, # frame index starts from 0
    frame_mask_left='/home/yxing/projects/stereo_PhysicalAttack/assets/masks/scene18_frame296_som_id5_mask.jpg.png',
)