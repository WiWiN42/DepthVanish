exp = dict(
    gpu=1,
    name='deeppruner_vkitti2_scene18_clone_frame296_id5', # model_dataset_scene_variation_frame_region
    round=100,
    n_checkpoint=10,
    save_dir='/data3/luqi/yxing/stereo_PhysicalAttack/results/temporal/base',
    log_level='debug',
)

dataset = dict(
    name='vkitti2',
    root='/data3/luqi/yxing/dataset/Virtual_KITTI2',
    scene='18',
    variation='clone',
    # normalize=True, # normalize to [0,1] range
)

model = dict(
    name='deeppruner',
    ckpt='/data3/luqi/yxing/stereo_PhysicalAttack/src/model/_checkpoints/DeepPruner-best-kitti.tar',
    loss = dict(
        unit_norm=True,
        alpha=0.1,
        beta=10,
        gamma=10,
        delta=1.0,
    )
)

optimizer = dict(
    lr=0.1
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
    frame_mask_left='/data3/luqi/yxing/stereo_PhysicalAttack/assets/masks/scene18_frame296_som_id5_mask.jpg.png',
)