#!/bin/bash
# =============================================================================
# Download KITTI Stereo 2015 (Scene Flow benchmark: stereo images + calibration)
# =============================================================================
#
# Used by this repo for:
#   - Default demo stereo pair in src/opt_case.py, src/opt_case_revise.py,
#     src/grid_opt.py, src/mask_opt.py, src/board_opt.py
#       training/image_2/000003_10.png   (left)
#       training/image_3/000003_10.png   (right)
#       training/calib_cam_to_cam/000003.txt
#   - The --dataset KITTI branch of src/opt_dataset.py / src/opt_dataset_revise.py /
#     src/eval_dataset.py (calib resolved per-sequence from training/calib_cam_to_cam/<seq>.txt)
#   - Calibration keys read by DigitalDeploy.load_calibrition() in src/utils/deploy.py:
#     P_rect_02, P_rect_03, R_rect_00
#   - Split lists: data/split/kitti_scene_flow_val_image0.txt,
#     kitti_scene_flow_val_image1.txt, kitti_scene_flow_val_disparity.txt
#     (the disparity list points at training/disp_occ_0/*.png)
#
# Source: KITTI Vision Benchmark Suite (Menze & Geiger, CVPR 2015)
#   https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
# License: CC BY-NC-SA 3.0 -- non-commercial research use only. By running this
#   script you agree to KITTI's terms of use as stated on the page above.
#
# Disk usage: ~2 GB images/disparity + ~2 MB calibration
#
# Usage: bash download_kitti_stereo2015.sh [TARGET_DIR]
#   TARGET_DIR defaults to /mnt/data/data_yxing/KITTI_stereo_2015, matching the
#   path baked into the scripts/defaults listed above.
# =============================================================================

set -euo pipefail

TARGET_DIR="${1:-/mnt/data/data_yxing/KITTI_stereo_2015}"
DOWNLOAD_DIR="${TARGET_DIR}/_downloads"

IMAGES_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"
CALIB_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip"

echo "==> Target directory: ${TARGET_DIR}"
mkdir -p "${DOWNLOAD_DIR}"

download() {
    local url="$1" out="$2"
    if [ -f "${out}" ]; then
        echo "==> Found existing $(basename "${out}"), skipping download"
    else
        echo "==> Downloading $(basename "${out}")"
        wget -c "${url}" -O "${out}"
    fi
}

download "${IMAGES_URL}" "${DOWNLOAD_DIR}/data_scene_flow.zip"
download "${CALIB_URL}" "${DOWNLOAD_DIR}/data_scene_flow_calib.zip"

echo "==> Extracting images + disparity (training/image_2, image_3, disp_occ_0, disp_noc_0, ...)"
unzip -oq "${DOWNLOAD_DIR}/data_scene_flow.zip" -d "${TARGET_DIR}"

echo "==> Extracting calibration (training/calib_cam_to_cam, testing/calib_cam_to_cam)"
unzip -oq "${DOWNLOAD_DIR}/data_scene_flow_calib.zip" -d "${TARGET_DIR}"

echo "==> Verifying files referenced by this repo's default arguments"
status=0
for f in \
    "${TARGET_DIR}/training/image_2/000003_10.png" \
    "${TARGET_DIR}/training/image_3/000003_10.png" \
    "${TARGET_DIR}/training/calib_cam_to_cam/000003.txt" \
    "${TARGET_DIR}/training/disp_occ_0/000003_10.png"
do
    if [ -f "${f}" ]; then
        echo "    OK      ${f}"
    else
        echo "    MISSING ${f}"
        status=1
    fi
done

echo
echo "==> Done. If TARGET_DIR differs from /mnt/data/data_yxing/KITTI_stereo_2015, update"
echo "    the --img_left/--img_right/--stereo_calib defaults in src/opt_case.py, src/grid_opt.py,"
echo "    src/mask_opt.py, src/board_opt.py and src/scripts/run_*.sh accordingly."
exit "${status}"
