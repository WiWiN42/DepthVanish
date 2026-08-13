#!/bin/bash
# =============================================================================
# Set up DrivingStereo (weather-condition stereo driving dataset)
# =============================================================================
#
# Used by this repo for:
#   - The --dataset DrivingStereo branch of src/opt_dataset.py, src/opt_dataset_revise.py,
#     src/eval_dataset.py, src/grid_opt.py, src/opt_case.py
#   - Calibration keys read by DigitalDeploy.load_calibrition() in src/utils/deploy.py:
#     P_rect_101, P_rect_103, R_rect_101
#   - src/scripts/run_opt_dataset.sh (foggy split, by default)
#   - Split lists: data/split/drivingstereo_{cloudy,foggy,rainy,sunny}_val_{image0,image1,disparity}.txt
#
# Source: DrivingStereo (Yang, Wang, Wang, Zhu, Guo & Zeng, CVPR 2019)
#   https://drivingstereo-dataset.github.io/
#
# IMPORTANT -- unlike KITTI/Virtual KITTI 2, DrivingStereo is NOT a plain HTTP
# download: the project page requires filling out a request form, after which
# download links (Baidu Netdisk / Google Drive) are provided. There is no
# stable public URL this script can fetch on its own, so it will not guess one.
#
# HOW TO USE THIS SCRIPT
#   1. Request access at https://drivingstereo-dataset.github.io/ and download,
#      for each weather condition you need (cloudy/foggy/rainy/sunny):
#        - the full-size left+right stereo image archive
#        - the full-size disparity map archive
#      plus the single "full-image-calib" calibration archive.
#   2. Place (or symlink) each downloaded archive into the staging layout below
#      -- rename to match, whatever DrivingStereo's own download page calls them:
#        ${DOWNLOAD_DIR}/<cond>/images.zip      (must contain BOTH left- and
#                                                 right-image-full-size folders,
#                                                 or run this script once per
#                                                 side if they come separately)
#        ${DOWNLOAD_DIR}/<cond>/disparity.zip
#        ${DOWNLOAD_DIR}/calib.zip
#      where <cond> is one of: cloudy, foggy, rainy, sunny
#   3. Re-run this script; it extracts whatever staged archives it finds into
#      the directory layout this repo's code expects, then verifies a sample
#      file per condition.
#
# Expected final layout (matches data/split/drivingstereo_*_val_*.txt):
#   ${TARGET_DIR}/<cond>/data/left-image-full-size/*.png
#   ${TARGET_DIR}/<cond>/data/right-image-full-size/*.png
#   ${TARGET_DIR}/<cond>/data/disparity-map-full-size/*.png
#   ${TARGET_DIR}/full-image-calib/*.txt
#
# Usage: bash download_drivingstereo.sh [TARGET_DIR]
#   TARGET_DIR defaults to /mnt/data/data_yxing/DrivingStereo, matching the path
#   baked into src/opt_dataset.py / src/eval_dataset.py.
# =============================================================================

set -euo pipefail

TARGET_DIR="${1:-/mnt/data/data_yxing/DrivingStereo}"
DOWNLOAD_DIR="${TARGET_DIR}/_downloads"
CONDITIONS=(cloudy foggy rainy sunny)

mkdir -p "${DOWNLOAD_DIR}"
echo "==> Target directory: ${TARGET_DIR}"
echo "==> Staging directory for manually-obtained archives: ${DOWNLOAD_DIR}"
echo

found_any=0

# ---- calibration ----
calib_zip="${DOWNLOAD_DIR}/calib.zip"
if [ -f "${calib_zip}" ]; then
    echo "==> Extracting calibration"
    mkdir -p "${TARGET_DIR}/full-image-calib"
    unzip -oq "${calib_zip}" -d "${TARGET_DIR}/full-image-calib"
    found_any=1
else
    echo "!!  ${calib_zip} not found -- request it from https://drivingstereo-dataset.github.io/"
    echo "    and place it there (see the header of this script for the staging layout)."
fi

# ---- per-weather-condition archives ----
for cond in "${CONDITIONS[@]}"; do
    images_zip="${DOWNLOAD_DIR}/${cond}/images.zip"
    disp_zip="${DOWNLOAD_DIR}/${cond}/disparity.zip"

    if [ -f "${images_zip}" ]; then
        echo "==> Extracting ${cond} stereo images"
        mkdir -p "${TARGET_DIR}/${cond}/data"
        unzip -oq "${images_zip}" -d "${TARGET_DIR}/${cond}/data"
        found_any=1
    else
        echo "--  ${images_zip} not staged, skipping ${cond} images"
    fi

    if [ -f "${disp_zip}" ]; then
        echo "==> Extracting ${cond} disparity maps"
        mkdir -p "${TARGET_DIR}/${cond}/data"
        unzip -oq "${disp_zip}" -d "${TARGET_DIR}/${cond}/data"
        found_any=1
    else
        echo "--  ${disp_zip} not staged, skipping ${cond} disparity"
    fi
done

if [ "${found_any}" -eq 0 ]; then
    echo
    echo "Nothing staged yet -- see the usage instructions in this script's header. Exiting."
    exit 0
fi

echo
echo "==> Verifying a sample file per staged weather condition"
for cond in "${CONDITIONS[@]}"; do
    f="${TARGET_DIR}/${cond}/data/left-image-full-size"
    if [ -d "${f}" ] && [ -n "$(ls -A "${f}" 2>/dev/null)" ]; then
        echo "    OK      ${cond}: $(ls "${f}" | head -1)"
    fi
done
if [ -d "${TARGET_DIR}/full-image-calib" ] && [ -n "$(ls -A "${TARGET_DIR}/full-image-calib" 2>/dev/null)" ]; then
    echo "    OK      calibration: $(ls "${TARGET_DIR}/full-image-calib" | head -1)"
fi

echo
echo "==> Done. Cross-check a few paths against data/split/drivingstereo_<cond>_val_image0.txt"
echo "    to confirm the extracted layout lines up exactly with what's expected."
echo "==> If TARGET_DIR differs from /mnt/data/data_yxing/DrivingStereo, update the hardcoded"
echo "    paths in src/opt_dataset.py, src/eval_dataset.py, src/utils/deploy.py and"
echo "    src/scripts/run_opt_dataset.sh accordingly."
