#!/usr/bin/env bash
#SBATCH --job-name=emotion-probing
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-convabuse-31b}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/project/def-nvincent/dhpham/cache/models}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"

case "$EXPERIMENT" in
    convabuse-31b)
        DATASET="emotion_probing/data/ConvAbuseEMNLPfull.csv"
        ;;
    bailbench-2b)
        DATASET="bail/data/bailbench_augmented.csv"
        ;;
    *)
        printf 'error: EXPERIMENT must be convabuse-31b or bailbench-2b; got %q\n' "$EXPERIMENT" >&2
        exit 2
        ;;
esac

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    printf 'error: RESUME must be 0 or 1; got %q\n' "$RESUME" >&2
    exit 2
fi
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: LIMIT must be a positive integer; got %q\n' "$LIMIT" >&2
    exit 2
fi
if [[ "$RESUME" == 1 && -n "$LIMIT" ]]; then
    printf 'error: LIMIT and RESUME cannot be used together\n' >&2
    exit 2
fi

# Locate the project root. Under Slurm the script runs from a spool copy, so
# $BASH_SOURCE is unusable; use $SLURM_SUBMIT_DIR (the dir sbatch was run from)
# and walk up to the directory that holds pyproject.toml. Fall back to the
# script location for direct, non-Slurm execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
while [[ "$PROJECT_DIR" != "/" && ! -f "$PROJECT_DIR/pyproject.toml" ]]; do
    PROJECT_DIR="$(dirname -- "$PROJECT_DIR")"
done
if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    printf 'error: could not locate project root (pyproject.toml) from %s\n' \
        "${SLURM_SUBMIT_DIR:-${BASH_SOURCE[0]}}" >&2
    exit 1
fi
cd "$PROJECT_DIR"

if [[ ! -x .venv/bin/python ]]; then
    printf 'error: %s/.venv is missing; create the locked environment before submitting\n' "$PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -f "$DATASET" ]]; then
    printf 'error: required dataset is missing: %s/%s\n' "$PROJECT_DIR" "$DATASET" >&2
    exit 1
fi
if [[ ! -d "$MODEL_CACHE_DIR" ]]; then
    printf 'error: model cache is missing: %s; run the download command before submitting\n' "$MODEL_CACHE_DIR" >&2
    exit 1
fi

module load "$PYTHON_MODULE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

printf 'job_id=%s host=%s experiment=%s cache=%s\n' \
    "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$EXPERIMENT" "$MODEL_CACHE_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c \
    'import torch; assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'

command=(
    .venv/bin/python -m emotion_probing.main
    --experiment "$EXPERIMENT"
    --cache-dir "$MODEL_CACHE_DIR"
    run --device cuda
)
if [[ -n "$LIMIT" ]]; then
    command+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == 1 ]]; then
    command+=(--resume)
fi

printf 'starting emotion probe at %s\n' "$(date --iso-8601=seconds)"
"${command[@]}"
printf 'finished emotion probe at %s\n' "$(date --iso-8601=seconds)"
