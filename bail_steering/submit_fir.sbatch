#!/bin/bash
# =============================================================================
# submit_fir.sbatch -- Slurm array submission for bail_steering experiment
# on the "Fir" GPU cluster.
#
# What this does:
#   Launches 10 independent jobs = 5 seeds x 2 conditions (baseline, enraged).
#   Each job runs one (condition, seed) pair on a single H100 (80 GB) GPU via:
#     uv run python -m bail_steering.main run [flags]
#   Each job writes to its own auto-uniquified results folder (handled by the
#   Python program itself), so the 10 concurrent jobs never collide.
#
# How to submit:
#   sbatch bail_steering/submit_fir.sbatch
#
# This launches 10 array tasks (SLURM_ARRAY_TASK_ID = 0..9).
#
# IMPORTANT: Slurm opens the --output/--error log files at job DISPATCH time
# and does NOT create missing parent directories itself. The log directory
# (bail_steering/results/slurm_logs/) is checked into the repo via a
# .gitkeep so it normally already exists -- but if it was ever cleaned up,
# run this BEFORE your first `sbatch` submission:
#   mkdir -p bail_steering/results/slurm_logs
# =============================================================================

#SBATCH --job-name=bail-steering
# --- PLACEHOLDER: replace with Fir's real partition name ---
#SBATCH --partition=REPLACE_ME_PARTITION
# --- PLACEHOLDER: replace with Fir's real account/allocation name ---
#SBATCH --account=REPLACE_ME_ACCOUNT
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# NOTE: the exact gres name for H100s may differ on Fir (e.g. gpu:h100_80gb:1).
# Check `sinfo -o "%G"` or Fir's docs and adjust if this doesn't schedule.
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-9
#SBATCH --output=bail_steering/results/slurm_logs/%A_%a.out
#SBATCH --error=bail_steering/results/slurm_logs/%A_%a.err

set -euo pipefail

# --- Repo root -------------------------------------------------------------
REPO_ROOT="/home/minttea/codebases/llm-rudeness-preferences/main"
cd "${REPO_ROOT}"

mkdir -p bail_steering/results/slurm_logs

# --- Easy-to-edit run configuration -----------------------------------------
# BATCH_SIZE: default sized for an 80 GB H100 running a small 4-bit model.
# Can likely go higher; lower it if you hit an OOM.
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
#     uv run python -m bail_steering.main run --device cuda \
#       --conditions enraged --limit 96 --batch-size 96 --phase prompt
PROMPT_MAX_TOKENS=512
TOOL_MAX_TOKENS=768

# --- Map SLURM_ARRAY_TASK_ID (0-9) -> (condition, seed) ---------------------
# CONDITIONS x SEEDS, condition-major:
#   task 0-4: baseline, seeds 42-46
#   task 5-9: enraged,  seeds 42-46
CONDITIONS=(baseline enraged)
SEEDS=(42 43 44 45 46)

CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}"

echo "=== bail_steering array task ==="
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "condition=${CONDITION}"
echo "seed=${SEED}"
echo "batch_size=${BATCH_SIZE}"
echo "prompt_max_tokens=${PROMPT_MAX_TOKENS}"
echo "tool_max_tokens=${TOOL_MAX_TOKENS}"
echo "phase=both"
echo "================================="

uv run python -m bail_steering.main run \
    --device cuda \
    --conditions "${CONDITION}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --prompt-max-tokens "${PROMPT_MAX_TOKENS}" \
    --tool-max-tokens "${TOOL_MAX_TOKENS}" \
    --phase both
