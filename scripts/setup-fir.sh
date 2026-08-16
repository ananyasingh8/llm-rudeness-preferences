#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup-fir.sh [--allow-pypi]

Create the project's .venv on Alliance Fir using the wheelhouse configured by
the active Python module. By default, installation fails if an exact locked
dependency is unavailable from DRAC. Pass --allow-pypi on a login node to use
PyPI as a fallback for missing wheels.

Environment variables:
  PYTHON_MODULE  Python module to load (default: python/3.12)
  UV_INSTALL_DIR User-local uv binary directory (default: $HOME/.local/bin)
  UV_CACHE_DIR   uv download cache (default: $SCRATCH/.cache/uv)
EOF
}

ALLOW_PYPI=0
case "${1:-}" in
    "") ;;
    --allow-pypi) ALLOW_PYPI=1 ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
if [[ -z "${UV_CACHE_DIR:-}" ]]; then
    if [[ -n "${SCRATCH:-}" ]]; then
        UV_CACHE_DIR="$SCRATCH/.cache/uv"
    else
        UV_CACHE_DIR="$HOME/.cache/uv"
    fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if ! type module >/dev/null 2>&1; then
    printf 'error: the module command is unavailable; run this script on a DRAC login node\n' >&2
    exit 1
fi
module load "$PYTHON_MODULE"

if ! command -v curl >/dev/null 2>&1; then
    printf 'error: curl is required to install uv in user space\n' >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    mkdir -p "$UV_INSTALL_DIR"
    printf 'Installing uv into %s\n' "$UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh |
        env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"
export UV_CACHE_DIR
export UV_NO_MANAGED_PYTHON=1
mkdir -p "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
    printf 'error: uv was not found after installation; expected it under %s\n' "$UV_INSTALL_DIR" >&2
    exit 1
fi

# DRAC exposes the wheelhouse under install/download/wheel find-links, not global.
find_links_output=""
for key in install.find-links download.find-links wheel.find-links; do
    if value="$(python -m pip config get "$key" 2>/dev/null)"; then
        find_links_output="$value"
        break
    fi
done
if [[ -z "${find_links_output//[[:space:]]/}" ]]; then
    printf 'error: could not read DRAC wheelhouse find-links from pip config (%s)\n' \
        "${PIP_CONFIG_FILE:-unset}" >&2
    exit 1
fi
find_links_output="${find_links_output//$'\n'/ }"
read -r -a wheelhouses <<< "$find_links_output"
if (( ${#wheelhouses[@]} == 0 )); then
    printf 'error: DRAC wheelhouse discovery returned no paths\n' >&2
    exit 1
fi

printf 'Using Python: %s\n' "$(command -v python)"
printf 'Using uv: %s\n' "$(command -v uv)"
printf 'Using DRAC wheelhouses:\n'
printf '  %s\n' "${wheelhouses[@]}"

requirements_file="$(mktemp "${TMPDIR:-/tmp}/emotion-probing-requirements.XXXXXX.txt")"
trap 'rm -f "$requirements_file"' EXIT

uv export \
    --locked \
    --no-dev \
    --no-hashes \
    --no-emit-project \
    --output-file "$requirements_file"

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python "$(command -v python)" --no-managed-python .venv
fi

sync_args=(
    uv pip sync
    --python .venv/bin/python
)
if [[ "$ALLOW_PYPI" == 0 ]]; then
    sync_args+=(--no-index)
fi
for wheelhouse in "${wheelhouses[@]}"; do
    sync_args+=(--find-links "$wheelhouse")
done
sync_args+=("$requirements_file")

"${sync_args[@]}"

printf '\nFir environment ready: %s/.venv\n' "$PROJECT_DIR"
printf 'Submit a smoke test with: LIMIT=10 sbatch emotion_probing/fir.slurm\n'
