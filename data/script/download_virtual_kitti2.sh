#!/bin/bash
# =============================================================================
# Download Virtual KITTI 2 (synthetic scenes: RGB, depth, optical flow, camera params)
# =============================================================================
#
# Used by this repo for:
#   - src/utils/dataset.py: VirtualKITTI2Loader (rgb, depth, forward/backward flow,
#     extrinsic.txt per scene/variation)
#   - The temporal/physical video attack: src/opt_video_case.py, src/opt_video_case_ori.py
#   - dataset.root in every config under src/config/temporal/*.py
#
# This repo only ever loads scenes '01', '02', '06', '18', '20' (see SCENE_ID in
# src/utils/dataset.py), but Virtual KITTI 2 is distributed as one archive PER
# DATA TYPE containing all scenes/variations bundled together -- there is no
# per-scene download, so the archives below are fetched and extracted in full.
#
# Source: Virtual KITTI 2 (Cabon, Murray & Humenberger, Naver Labs Europe, 2020)
#   https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
# License: CC BY-NC-SA 3.0 -- non-commercial research use only.
#
# Disk usage: roughly 40 GB total (rgb + depth + forward/backward flow + textgt).
# Segmentation/scene-flow archives are intentionally NOT downloaded here --
# nothing in VirtualKITTI2Loader.file_templates reads them (see the commented-out
# "segmentation"/"instance" entries in src/utils/dataset.py).
#
# Usage: bash download_virtual_kitti2.sh [TARGET_DIR]
#   TARGET_DIR defaults to /mnt/data/data_yxing/Virtual_KITTI2, matching
#   dataset.root in src/config/temporal/*.py.
# =============================================================================

set -euo pipefail

TARGET_DIR="${1:-/mnt/data/data_yxing/Virtual_KITTI2}"
DOWNLOAD_DIR="${TARGET_DIR}/_downloads"
BASE_URL="http://download.europe.naverlabs.com/virtual_kitti_2.2.0"

ARCHIVES=(
    "vkitti_2.2.0_rgb.tar"
    "vkitti_2.2.0_depth.tar"
    "vkitti_2.2.0_forwardFlow.tar"
    "vkitti_2.2.0_backwardFlow.tar"
    "vkitti_2.2.0_textgt.tar.gz"
)

echo "==> Target directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}" "${DOWNLOAD_DIR}"

download() {
    local name="$1" out="${DOWNLOAD_DIR}/$1"
    if [ -f "${out}" ]; then
        echo "==> Found existing ${name}, skipping download"
    else
        echo "==> Downloading ${name}"
        wget -c "${BASE_URL}/${name}" -O "${out}"
    fi
}

for name in "${ARCHIVES[@]}"; do
    download "${name}"
done

echo "==> Extracting into ${TARGET_DIR} (creates Scene<NN>/<variation>/... per archive)"
for name in "${ARCHIVES[@]}"; do
    echo "    ${name}"
    tar -xf "${DOWNLOAD_DIR}/${name}" -C "${TARGET_DIR}"
done

echo "==> Verifying Scene18/clone (the scene/variation used by src/config/temporal/*.py defaults)"
status=0
for f in \
    "${TARGET_DIR}/Scene18/clone/frames/rgb/Camera_0/rgb_00000.jpg" \
    "${TARGET_DIR}/Scene18/clone/frames/depth/Camera_0/depth_00000.png" \
    "${TARGET_DIR}/Scene18/clone/frames/forwardFlow/Camera_0/flow_00000.png" \
    "${TARGET_DIR}/Scene18/clone/frames/backwardFlow/Camera_0/backwardFlow_00000.png" \
    "${TARGET_DIR}/Scene18/clone/extrinsic.txt"
do
    if [ -f "${f}" ]; then
        echo "    OK      ${f}"
    else
        echo "    MISSING ${f}"
        status=1
    fi
done

echo
echo "==> Done. If TARGET_DIR differs from /mnt/data/data_yxing/Virtual_KITTI2, update"
echo "    dataset.root in every src/config/temporal/*.py config."
exit "${status}"
