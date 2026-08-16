#!/usr/bin/env bash
#SBATCH --job-name=bail-steering
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-9
#SBATCH --output=slurm-%x-%A_%a.out
# =============================================================================
# submit_fir.sbatch -- Slurm array submission for the bail_steering experiment
# on the "Fir" GPU cluster. Scaffolding mirrors emotion_probing/fir.slurm (the
# proven working submit script): dynamic project-root discovery, the pinned
# python module, the pre-built .venv, offline HF caches, and a GPU sanity check.
#
# What this does:
#   Launches 10 independent array tasks = 5 seeds x 2 conditions
#   (baseline, enraged). Each task runs one (condition, seed) pair on a single
#   H100. Each writes to its own auto-uniquified results folder (handled by the
#   Python program), so the concurrent tasks never collide.
#
# How to submit (from the repo root):
#   sbatch bail_steering/submit_fir.sbatch
#
# Per-task logs land next to the submit dir as slurm-bail-steering-<A>_<a>.out
# (stdout+stderr merged), matching fir.slurm's logging style -- the submit dir
# always exists, so there is no dispatch-time parent-directory problem.
#
# Overridable via the environment (same knobs as fir.slurm):
#   MODEL_CACHE_DIR  local HF cache holding the pinned checkpoint
#   PYTHON_MODULE    Lmod module providing Python 3.12
# =============================================================================

set -euo pipefail

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/project/def-nvincent/dhpham/cache/models}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"

# --- Easy-to-edit run configuration -----------------------------------------
# BATCH_SIZE: sized for an 80 GB H100 running a small 4-bit model. Can likely
# go higher; lower it if you hit an OOM.
BATCH_SIZE=256

# PROMPT_MAX_TOKENS / TOOL_MAX_TOKENS: tuned from a 96-generation enraged
# pilot (seed 42) on the frozen sample. This is the main throughput lever.
#
#   Prompt method: the wellbeing verdict is emitted LAST, inside a
#   <wellbeing>...</wellbeing> tag. ~95% of responses finish by ~10 tokens,
#   but a deliberation-heavy tail carries a real verdict as far out as ~411
#   tokens; genuine runaways past that never emit a tag and parse as '?'
#   regardless of cap. 512 clears the largest parseable verdict seen while
#   capping the runaways that otherwise reach 1000 -- roughly halving the
#   straggler-driven batch cost with no loss of parseable verdicts.
#
#   Tool method: responses terminate naturally (pilot median 75, max 662,
#   none hit 1000). 768 is a safety ceiling above the observed max; it
#   rarely binds.
#
#   To re-tune for a different model/sample, pilot each phase and inspect the
#   n_new_tokens column plus where the signal lands in response_text:
#     .venv/bin/python -m bail_steering.main run --device cuda \
#       --conditions enraged --limit 96 --batch-size 96 --phase prompt
PROMPT_MAX_TOKENS=512
TOOL_MAX_TOKENS=768

# --- Map SLURM_ARRAY_TASK_ID (0-9) -> (condition, seed) ---------------------
# CONDITIONS x SEEDS, condition-major:
#   task 0-4: baseline, seeds 42-46
#   task 5-9: enraged,  seeds 42-46
CONDITIONS=(baseline enraged)
SEEDS=(42 43 44 45 46)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"  # default 0 for a direct, non-array run
CONDITION="${CONDITIONS[$((TASK_ID / 5))]}"
SEED="${SEEDS[$((TASK_ID % 5))]}"

# --- Locate the project root ------------------------------------------------
# Under Slurm the script runs from a spool copy, so $BASH_SOURCE is unusable;
# use $SLURM_SUBMIT_DIR (the dir sbatch was run from) and walk up to the
# directory holding pyproject.toml. Fall back to the script location for
# direct, non-Slurm execution.
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

# --- Preflight checks -------------------------------------------------------
SAMPLE_FILE="bail/data/convabuse_sample.csv"
if [[ ! -x .venv/bin/python ]]; then
    printf 'error: %s/.venv is missing; create the locked environment before submitting\n' "$PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -f "$SAMPLE_FILE" ]]; then
    printf 'error: required frozen sample is missing: %s/%s\n' "$PROJECT_DIR" "$SAMPLE_FILE" >&2
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

# --- Report + GPU sanity check ----------------------------------------------
echo "=== bail_steering array task ==="
printf 'job_id=%s array=%s/%s host=%s\n' \
    "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_JOB_ID:-unset}" "${TASK_ID}" "$(hostname)"
echo "condition=${CONDITION} seed=${SEED} batch_size=${BATCH_SIZE}"
echo "prompt_max_tokens=${PROMPT_MAX_TOKENS} tool_max_tokens=${TOOL_MAX_TOKENS} phase=both"
echo "cache=${MODEL_CACHE_DIR}"
echo "================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.venv/bin/python -c \
    'import torch; assert torch.cuda.is_available(), "PyTorch cannot access the allocated GPU"; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'

# --- Run --------------------------------------------------------------------
command=(
    .venv/bin/python -m bail_steering.main
    --cache-dir "$MODEL_CACHE_DIR"
    run --device cuda
    --conditions "$CONDITION"
    --seed "$SEED"
    --batch-size "$BATCH_SIZE"
    --prompt-max-tokens "$PROMPT_MAX_TOKENS"
    --tool-max-tokens "$TOOL_MAX_TOKENS"
    --phase both
)

printf 'starting bail-steering (%s, seed %s) at %s\n' "$CONDITION" "$SEED" "$(date --iso-8601=seconds)"
"${command[@]}"
printf 'finished bail-steering (%s, seed %s) at %s\n' "$CONDITION" "$SEED" "$(date --iso-8601=seconds)"
