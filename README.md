
# Dataset & Metric for Stereo Depth

## dataset 
- KITTI 2015
- FlyingThings3D

## metric
- D1-all error (KITTI official)
- end-point-error (EPE) (official scene flow FlyingThings)

# Dataset & Metric for Mono Depth

## dataset
- KITTI depth
- NYU Depth v2 (640 × 480)

## metric
- mean depth estimation error
- ratio of the affected region


# 运行指令速查

## 1. 唤醒 dpa 虚拟环境

```bash
source /home/luqi/anaconda3/etc/profile.d/conda.sh
conda activate dpa
```

## 2. 运行视频深度消失攻击

```bash
cd /data3/luqi/yxing/stereo_PhysicalAttack
python src/opt_video_case.py --cfg src/config/temporal/aanet.py
```

运行结果输出到 `results/temporal/distance/aanet_vkitti2_scene18_clone_frame296_id5_test/`（名字来自 config 里的 `exp.name`）。

---

### 后台运行（推荐，避免挂断终端）

```bash
cd /data3/luqi/yxing/stereo_PhysicalAttack
nohup python src/opt_video_case.py --cfg src/config/temporal/aanet.py > run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

cd /data3/luqi/yxing/stereo_PhysicalAttack
nohup python src/opt_video_case.py --cfg src/config/temporal/psmnet.py > run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

日志实时查看：

```bash
tail -f run.log
```
