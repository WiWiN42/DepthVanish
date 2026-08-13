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
# CAUTION: the exact S3 URL layout below (raw_data/<drive>/<drive>.zip) matches
# the pattern used by several public KITTI download scripts at the time this
# was written, but KITTI has changed its hosting before. If a download 404s,
# check https://www.cvlibs.net/datasets/kitti/raw_data.php for the current
# per-sequence download links and either update BASE_URL or fetch that one
# sequence by hand.
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
    echo "     downloads than '_sync' ones; same URL convention is assumed for both, see the"
    echo "     CAUTION note above if any of these specifically fail to download)"
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

    drive_out="${DOWNLOAD_DIR}/${drive}.zip"
    if download "${BASE_URL}/${drive}/${drive}.zip" "${drive_out}"; then
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
    echo "    ${fail_count} download(s) failed -- see the CAUTION note in this script's header"
    echo "    and retry those sequences manually from https://www.cvlibs.net/datasets/kitti/raw_data.php"
fi
echo "==> If TARGET_DIR differs from /mnt/data/data_yxing/KITTI_raw, update the paths recorded in"
echo "    data/split/kitti_train_image0.txt and data/split/kitti_train_image1.txt to match."
