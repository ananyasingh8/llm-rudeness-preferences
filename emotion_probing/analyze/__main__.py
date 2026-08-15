"""CLI entry point: `uv run python -m emotion_probing.analyze [--run PATH]`."""

from __future__ import annotations

import argparse
import json
import sys

from emotion_probing.analyze.bailbench import analyze_bailbench
from emotion_probing.analyze.common import find_run_dir
from emotion_probing.analyze.convabuse import analyze_convabuse


def main() -> int:
    """Analyze one run folder (the latest by default)."""
    parser = argparse.ArgumentParser(description="Analyze a probing run.")
    parser.add_argument(
        "--run",
        default=None,
        help="run folder to analyze (default: the latest under results/)",
    )
    args = parser.parse_args()
    run_dir = find_run_dir(args.run)
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    print(f"Analyzing {run_dir.name} ({run_info['dataset']})\n")
    if run_info["dataset"] == "bailbench":
        analyze_bailbench(run_dir)
    elif run_info["dataset"] == "convabuse":
        analyze_convabuse(run_dir)
    else:
        print(f"error: unknown dataset {run_info['dataset']!r}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
