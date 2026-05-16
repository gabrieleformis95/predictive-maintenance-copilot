#!/usr/bin/env bash
# Download the NASA C-MAPSS turbofan degradation dataset.
#
# The original NASA PCoE site sometimes blocks scripted downloads; this script
# falls back to the public Kaggle mirror as long as the Kaggle CLI is set up.
# Manual fallback: download CMAPSSData.zip from any public mirror and unzip
# into data/raw/CMAPSSData/.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
TARGET_DIR="${RAW_DIR}/CMAPSSData"

mkdir -p "${RAW_DIR}"

if [[ -f "${TARGET_DIR}/train_FD001.txt" ]]; then
  echo "✔ C-MAPSS already present at ${TARGET_DIR} — skipping."
  exit 0
fi

echo "→ Downloading NASA C-MAPSS dataset…"

# Try Kaggle first (most reliable mirror).
if command -v kaggle >/dev/null 2>&1; then
  echo "  Using Kaggle CLI…"
  kaggle datasets download -d behrad3d/nasa-cmaps -p "${RAW_DIR}" --unzip
elif command -v curl >/dev/null 2>&1; then
  echo "  Kaggle CLI not found. Please download CMAPSSData.zip manually:"
  echo
  echo "    1. https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
  echo "       (or https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data)"
  echo "    2. Unzip into:   ${TARGET_DIR}"
  echo
  exit 1
else
  echo "Neither kaggle nor curl available. Install one of them, then re-run."
  exit 1
fi

# Re-arrange if the zip extracted a different layout.
if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "→ Normalizing directory layout into ${TARGET_DIR}"
  mkdir -p "${TARGET_DIR}"
  find "${RAW_DIR}" -maxdepth 2 -name "train_FD00*.txt" -exec mv {} "${TARGET_DIR}/" \;
  find "${RAW_DIR}" -maxdepth 2 -name "test_FD00*.txt"  -exec mv {} "${TARGET_DIR}/" \;
  find "${RAW_DIR}" -maxdepth 2 -name "RUL_FD00*.txt"   -exec mv {} "${TARGET_DIR}/" \;
fi

echo "✔ Done."
ls -1 "${TARGET_DIR}"
