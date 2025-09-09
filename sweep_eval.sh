#!/usr/bin/env bash
set -euo pipefail

# Defaults (change if needed)
DATASET="bc5cdr"
ENT_TYPE="Disease"
KB="medic"
THRESHOLD="0.15"
SAVE_ROOT="test/test_data/saved_trees"   # where xmodel_* folders live
PY_CMD="python"                           # or python3

# Accept optional args:
#   -m|--model-dir  : explicit model dir (overrides auto-detect)
#   -r|--root       : base dir that contains xmodel_* folders
#   -d|--dataset    : dataset name
#   --ent-type      : entity type
#   --kb            : kb name
MODEL_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model-dir) MODEL_DIR="$2"; shift 2;;
    -r|--root) SAVE_ROOT="$2"; shift 2;;
    -d|--dataset) DATASET="$2"; shift 2;;
    --ent-type) ENT_TYPE="$2"; shift 2;;
    --kb) KB="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# Auto-detect most recent xmodel_* directory if none provided
if [[ -z "${MODEL_DIR}" ]]; then
  if compgen -G "${SAVE_ROOT}/xmodel_*" >/dev/null; then
    # newest by modification time
    MODEL_DIR="$(ls -dt "${SAVE_ROOT}"/xmodel_* | head -1)"
  else
    echo "No xmodel_* directories found under ${SAVE_ROOT}. Provide --model-dir." >&2
    exit 1
  fi
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model dir not found: ${MODEL_DIR}" >&2
  exit 1
fi

echo "Using model_dir: ${MODEL_DIR}"
echo "Dataset=${DATASET}  EntType=${ENT_TYPE}  KB=${KB}"
echo

BEAMS=(1 5 10 15 20 25 30 40 50)
TOPKS=(1 5 10 50 100 200 500)

for b in "${BEAMS[@]}"; do
  for k in "${TOPKS[@]}"; do
    echo ">>> Running: -beam_size ${b} -top_k ${k}"
    ${PY_CMD} src/python/xlinker/evaluate_testing.py \
      -dataset "${DATASET}" \
      -ent_type "${ENT_TYPE}" \
      -kb "${KB}" \
      -model_dir "${MODEL_DIR}" \
      -beam_size "${b}" \
      -top_k "${k}" \
      --abbrv --pipeline --threshold "${THRESHOLD}" --ppr
    echo
  done
done

echo "All runs completed."