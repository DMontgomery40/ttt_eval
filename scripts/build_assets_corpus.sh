#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="${ROOT_DIR}/assets"
OUT_PATH="${ROOT_DIR}/training_data/assets_papers.txt"

mkdir -p "${ROOT_DIR}/training_data"

if ! command -v pdftotext >/dev/null 2>&1; then
  echo "[assets_corpus] ERROR: pdftotext not found" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

{
  echo "# assets/ PDF corpus"
  echo "# generated_at_unix=$(date +%s)"
  echo
} > "${OUT_PATH}"

shopt -s nullglob
pdfs=("${ASSETS_DIR}"/*.pdf)
if [[ ${#pdfs[@]} -eq 0 ]]; then
  echo "[assets_corpus] No PDFs found in ${ASSETS_DIR}" >&2
  exit 1
fi

for pdf in "${pdfs[@]}"; do
  base="$(basename "${pdf}")"
  echo "[assets_corpus] ${base}"

  txt="${tmp_dir}/${base}.txt"
  pdftotext -layout "${pdf}" "${txt}" >/dev/null 2>&1 || true

  {
    echo
    echo "=============================="
    echo "FILE: assets/${base}"
    echo "=============================="
    echo
    if [[ -s "${txt}" ]]; then
      cat "${txt}"
    else
      echo "[assets_corpus] WARNING: no text extracted"
    fi
    echo
  } >> "${OUT_PATH}"
done

echo "[assets_corpus] Wrote ${OUT_PATH}"

