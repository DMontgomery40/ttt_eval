#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT_DIR/artifacts}"
NANO_API_HOST="${NANO_API_HOST:-127.0.0.1}"
NANO_API_PORT="${NANO_API_PORT:-13579}"

DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-5173}"

pick_python() {
  local candidates=()
  if [[ -n "${PYTHON:-}" ]]; then candidates+=("${PYTHON}"); fi
  # Prefer whatever the user has on PATH first.
  # (pyenv may not expose a `python3.11` shim by default.)
  candidates+=("python3" "python")

  for py in "${candidates[@]}"; do
    if ! command -v "$py" >/dev/null 2>&1; then
      continue
    fi
    if "$py" -c "import torch, fastapi, uvicorn, pydantic" >/dev/null 2>&1; then
      echo "$py"
      return 0
    fi
  done

  return 1
}

PY_BIN="$(pick_python || true)"
if [[ -z "${PY_BIN}" ]]; then
  # If pyenv is installed, try installed interpreters directly.
  if command -v pyenv >/dev/null 2>&1; then
    PYENV_ROOT="$(pyenv root 2>/dev/null || true)"
    if [[ -n "${PYENV_ROOT}" ]]; then
      while IFS= read -r ver; do
        [[ -z "${ver}" ]] && continue
        [[ "${ver}" == "system" ]] && continue

        candidate="${PYENV_ROOT}/versions/${ver}/bin/python"
        if [[ -x "${candidate}" ]] && "${candidate}" -c "import torch, fastapi, uvicorn, pydantic" >/dev/null 2>&1; then
          PY_BIN="${candidate}"
          break
        fi
      done < <(pyenv versions --bare 2>/dev/null || true)
    fi
  fi
fi

if [[ -z "${PY_BIN}" ]]; then
  echo "[start] Could not find a Python with torch + fastapi + uvicorn + pydantic available."
  echo "[start] Try (example):"
  if command -v pyenv >/dev/null 2>&1; then
    echo "  pyenv shell 3.11.7"
    echo "  python -m pip install -e ."
    echo "  python -m pip install fastapi uvicorn pydantic"
  else
    echo "  python -m pip install -e ."
    echo "  python -m pip install fastapi uvicorn pydantic"
  fi
  exit 1
fi

if [[ ! -f "${ARTIFACTS_ROOT}/base/base_checkpoint.pt" ]]; then
  echo "[start] No artifacts found at ${ARTIFACTS_ROOT}. Generating a tiny demo..."
  "${PY_BIN}" "${ROOT_DIR}/ttt_ssm_nano/phase1_branching_muon.py" \
    --artifacts_root "${ARTIFACTS_ROOT}" \
    init_base \
    --pretrain_steps 50 \
    --pretrain_batch 8 \
    --pretrain_seq 16 \
    --seed 1 \
    --env_mode linear \
    --u_dim 16 \
    --n_state 32 \
    --lr 0.005 \
    --momentum 0.95 \
    --ns_steps 2 \
    --adjust_lr_fn none

  "${PY_BIN}" "${ROOT_DIR}/ttt_ssm_nano/phase1_branching_muon.py" \
    --artifacts_root "${ARTIFACTS_ROOT}" \
    new_session \
    --session_id demo_s1 \
    --mu 0.12 \
    --env_mode linear \
    --seed 1 \
    --lr 0.005 \
    --momentum 0.95 \
    --ns_steps 2 \
    --chunk 32 \
    --buffer_len 32 \
    --rollback_tol 0.20 \
    --grad_norm_max 20 \
    --state_norm_max 1000000 \
    --adjust_lr_fn none

  "${PY_BIN}" "${ROOT_DIR}/ttt_ssm_nano/phase1_branching_muon.py" \
    --artifacts_root "${ARTIFACTS_ROOT}" \
    run_session \
    --session_id demo_s1 \
    --steps 128 \
    --seed 2
fi

echo "[start] Starting artifacts API: http://${NANO_API_HOST}:${NANO_API_PORT}"
"${PY_BIN}" -m ttt_ssm_nano.artifacts_api \
  --artifacts_root "${ARTIFACTS_ROOT}" \
  --host "${NANO_API_HOST}" \
  --port "${NANO_API_PORT}" &
API_PID=$!

cleanup() {
  if kill -0 "${API_PID}" >/dev/null 2>&1; then
    kill "${API_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ ! -d "${ROOT_DIR}/dashboard/node_modules" ]]; then
  echo "[start] Installing dashboard dependencies..."
  npm -C "${ROOT_DIR}/dashboard" install
fi

echo "[start] Starting dashboard: http://${DASHBOARD_HOST}:${DASHBOARD_PORT}"
VITE_NANO_API_URL="http://${NANO_API_HOST}:${NANO_API_PORT}" \
  npm -C "${ROOT_DIR}/dashboard" run dev -- --host "${DASHBOARD_HOST}" --port "${DASHBOARD_PORT}"
