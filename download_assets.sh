#!/usr/bin/env bash
set -euo pipefail

RECORD_ID="20566871"
BASE_URL="https://zenodo.org/api/records/${RECORD_ID}/files"
ARCHIVE="kraft-uavswarm-assets.tar.gz"
CHECKSUM="kraft-uavswarm-assets.sha256"

download_file() {
  local file="$1"
  local url="${BASE_URL}/${file}/content"

  if [[ -f "${file}" ]]; then
    echo "[skip] ${file} already exists"
    return
  fi

  echo "[download] ${file}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 5 -o "${file}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${file}" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

download_file "${CHECKSUM}"
download_file "${ARCHIVE}"

echo "[verify] ${ARCHIVE}"
sha256sum -c "${CHECKSUM}"

echo "[extract] ${ARCHIVE}"
tar -xzf "${ARCHIVE}"

echo "[done] Assets are available under dataset/, detections/, checkpoints/, and tracks/."
