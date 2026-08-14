#!/bin/bash
# =============================================================================
# Download KITTI Raw -- only the drive sequences referenced by
# data/split/kitti_train_image0.txt / kitti_train_image1.txt
# =============================================================================
#
# NOTE: unlike the other three scripts in this folder, kitti_train_image0.txt /
# kitti_train_image1.txt are NOT read as a default argument by any script
# currently in src/ -- they're a prepared split list already sitting in
# data/split/ (47,991 stereo pairs) that nothing in this repo's top-level
# pipeline (opt_case.py, opt_dataset.py, eval_*.py, opt_video_case.py, ...)
# consumes yet. This script exists because the split list is real and clearly
# intended for use; wire it in wherever needed (e.g. pass it as
# --img_left_file/--img_right_file to src/opt_dataset.py) once downloaded.
#
# KITTI Raw is NOT one archive -- it's one zip per drive sequence plus one
# calibration zip per capture date. This script parses the split file to find
# only the (date, drive) pairs actually referenced, rather than pulling the
# full ~180 GB dataset.
#
# Source: KITTI Raw Data (Geiger, Lenz, Stiller & Urtasun, IJRR 2013)
#   https://www.cvlibs.net/datasets/kitti/raw_data.php
# License: CC BY-NC-SA 3.0 -- non-commercial research use only.
#
# URL layout verified 2026-08-13 with curl -I (each returned 200):
#   calibration: raw_data/<date>_calib.zip
#   drive data:  raw_data/<date>_drive_<seq>/<date>_drive_<seq>_{sync,extract}.zip
#   (note the folder segment drops the _sync/_extract suffix that the filename keeps)
# If KITTI changes hosting again and a download 404s, check
# https://www.cvlibs.net/datasets/kitti/raw_data.php for current links.
#
# Usage: bash download_kitti_raw.sh [TARGET_DIR]
#   TARGET_DIR defaults to /mnt/data/data_yxing/KITTI_raw, matching the paths
#   already recorded in data/split/kitti_train_image0.txt / image1.txt.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT_FILE="${SCRIPT_DIR}/../split/kitti_train_image0.txt"
TARGET_DIR="${1:-/mnt/data/data_yxing/KITTI_raw}"
DOWNLOAD_DIR="${TARGET_DIR}/_downloads"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

[ -f "${SPLIT_FILE}" ] || { echo "Split file not found: ${SPLIT_FILE}" >&2; exit 1; }
mkdir -p "${TARGET_DIR}" "${DOWNLOAD_DIR}"

echo "==> Parsing ${SPLIT_FILE} for referenced (date, drive) pairs..."
# path shape: .../<date>/<date>_drive_<seq>_sync/image_02/data/<frame>.png
# A handful of drives in this split are "_extract" (unsynced+unrectified) rather
# than "_sync" (synced+rectified) -- both suffixes are matched here.
mapfile -t DRIVES < <(
    sed -E 's#.*/([0-9]{4}_[0-9]{2}_[0-9]{2})/([0-9]{4}_[0-9]{2}_[0-9]{2}_drive_[0-9]{4}_(sync|extract))/.*#\1 \2#' \
        "${SPLIT_FILE}" | sort -u
)
echo "==> Found ${#DRIVES[@]} unique drive sequence(s):"
printf '    %s\n' "${DRIVES[@]}"
n_extract=$(printf '%s\n' "${DRIVES[@]}" | grep -c "_extract$" || true)
if [ "${n_extract}" -gt 0 ]; then
    echo "    (${n_extract} of these are '_extract' unsynced sequences -- typically much larger"
    echo "     downloads than '_sync' ones)"
fi
echo

read -rp "Proceed with download? Each drive can be several GB. [y/N] " CONFIRM
case "${CONFIRM}" in
    y|Y) ;;
    *) echo "Aborted."; exit 0 ;;
esac

download() {
    local url="$1" out="$2"
    if [ -f "${out}" ]; then
        echo "==> Found existing $(basename "${out}"), skipping download"
        return 0
    fi
    echo "==> Downloading $(basename "${out}")"
    if ! wget -c "${url}" -O "${out}"; then
        echo "    !! failed: ${url}" >&2
        rm -f "${out}"
        return 1
    fi
}

DATES_SEEN=()
fail_count=0
for entry in "${DRIVES[@]}"; do
    date="${entry%% *}"
    drive="${entry##* }"

    if [[ ! " ${DATES_SEEN[*]:-} " =~ " ${date} " ]]; then
        DATES_SEEN+=("${date}")
        calib_out="${DOWNLOAD_DIR}/${date}_calib.zip"
        if download "${BASE_URL}/${date}_calib.zip" "${calib_out}"; then
            unzip -oq "${calib_out}" -d "${TARGET_DIR}" || echo "    !! extraction failed for ${date}_calib.zip" >&2
        else
            fail_count=$((fail_count + 1))
        fi
    fi

    # The URL's folder segment drops the _sync/_extract suffix; the filename keeps it.
    drive_dir="${drive%_sync}"
    drive_dir="${drive_dir%_extract}"
    drive_out="${DOWNLOAD_DIR}/${drive}.zip"
    if download "${BASE_URL}/${drive_dir}/${drive}.zip" "${drive_out}"; then
        # The zip's own top-level entry is already "<date>/", so extract straight
        # into TARGET_DIR -- extracting into TARGET_DIR/<date> would double-nest it
        # (TARGET_DIR/<date>/<date>/<date>_drive_..._sync/...), same as the calib zip above.
        unzip -oq "${drive_out}" -d "${TARGET_DIR}" || echo "    !! extraction failed for ${drive}.zip" >&2
    else
        fail_count=$((fail_count + 1))
    fi
done

echo
echo "==> Done. ${#DATES_SEEN[@]} date(s), ${#DRIVES[@]} drive(s) processed into ${TARGET_DIR}."
if [ "${fail_count}" -gt 0 ]; then
    echo "    ${fail_count} download(s) failed -- retry those sequences manually from"
    echo "    https://www.cvlibs.net/datasets/kitti/raw_data.php"
fi
echo "==> If TARGET_DIR differs from /mnt/data/data_yxing/KITTI_raw, update the paths recorded in"
echo "    data/split/kitti_train_image0.txt and data/split/kitti_train_image1.txt to match."
