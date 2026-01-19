#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT_DIR}/training_data/.sources/TinyStories"
LEGACY_DEST="${ROOT_DIR}/training_data/TinyStories"

mkdir -p "${ROOT_DIR}/training_data"
mkdir -p "$(dirname "${DEST}")"

if ! command -v git >/dev/null 2>&1; then
  echo "[tinystories] ERROR: git not found" >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs --version >/dev/null 2>&1; then
  echo "[tinystories] ERROR: git lfs not found" >&2
  exit 1
fi

git lfs install --local >/dev/null 2>&1 || true

# Back-compat: older runs cloned into training_data/TinyStories (not hidden),
# which the trainer may ingest if you train on training_data/. Move it under a
# hidden directory so it won't be treated as corpus by default.
if [[ -d "${LEGACY_DEST}/.git" && ! -d "${DEST}/.git" ]]; then
  echo "[tinystories] Moving legacy clone to hidden path: ${DEST}"
  rm -rf "${DEST}"
  mv "${LEGACY_DEST}" "${DEST}"
fi

if [[ ! -d "${DEST}/.git" ]]; then
  echo "[tinystories] Cloning (LFS skip-smudge) to ${DEST}"
  rm -rf "${DEST}"
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/roneneldan/TinyStories "${DEST}"
fi

echo "[tinystories] Pulling TinyStoriesV2-GPT4-valid.txt via LFS"
git -C "${DEST}" lfs pull -I "TinyStoriesV2-GPT4-valid.txt"

if [[ ! -f "${DEST}/TinyStoriesV2-GPT4-valid.txt" ]]; then
  echo "[tinystories] ERROR: TinyStoriesV2-GPT4-valid.txt not found after pull" >&2
  exit 1
fi

cp -f "${DEST}/TinyStoriesV2-GPT4-valid.txt" "${ROOT_DIR}/training_data/TinyStoriesV2-GPT4-valid.txt"
echo "[tinystories] Wrote training_data/TinyStoriesV2-GPT4-valid.txt"
