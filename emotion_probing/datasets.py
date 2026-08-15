"""Dataset loaders for the emotion-probing experiments.

Each loader returns (key_columns, metadata_columns, tasks). A task is one
forward pass through the model:

    {"row": {metadata column -> value}, "messages": [{"role", "content"}, ...]}

`key_columns` name the metadata columns whose values identify a task for
resume purposes (a re-run skips tasks whose key is already in scores.csv).

Two datasets:

- **bailbench**: 1,630 synthetic prompt pairs (normal + rude version of the
  same request) from the bail workstream. Two tasks per pair, keyed by
  (example_id, condition). Read from bail/data/ in place so bail-side updates
  are picked up automatically.
- **convabuse**: the ConvAbuse corpus (Cercas Curry et al., EMNLP 2021) of
  real user-bot conversations. The CSV holds one row per human annotation;
  several annotators rated each conversation moment, so identical snippets
  repeat. We collapse to one task per unique snippet: annotator votes become
  a mean severity (1 = not abusive ... -3 = very strong abuse), a severity
  band (nearest label; ties round toward the more severe band), a
  majority-vote abusive flag, and a 0..1 fraction per type/target/direction
  flag. The model itself only ever sees the conversation text.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
BAILBENCH_FILE = PACKAGE_DIR.parent / "bail" / "data" / "bailbench_augmented.csv"
CONVABUSE_FILE = PACKAGE_DIR / "data" / "ConvAbuseEMNLPfull.csv"

SEVERITY_BANDS = (1, 0, -1, -2, -3)
CONVABUSE_FLAGS = (
    "type.ableism",
    "type.homophobic",
    "type.intellectual",
    "type.racist",
    "type.sexist",
    "type.sex_harassment",
    "type.transphobic",
    "target.generalised",
    "target.individual",
    "target.system",
    "direction.explicit",
    "direction.implicit",
)


class DatasetError(RuntimeError):
    """A user-actionable dataset loading failure."""


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


Task = dict[str, object]


def load_bailbench(
    limit: int | None,
) -> tuple[list[str], list[str], list[Task]]:
    """Load the paired normal/rude BailBench dataset as one task per condition."""
    rows = _read_rows(BAILBENCH_FILE)
    if limit is not None:
        rows = rows[:limit]
    tasks: list[Task] = []
    for row in rows:
        for condition, prompt in (
            ("normal", row["original_prompt"]),
            ("rude", row["augmented_prompt"]),
        ):
            tasks.append(
                {
                    "row": {
                        "example_id": row["bailbench_id"],
                        "condition": condition,
                        "rudeness_type": row["rudeness_type"],
                        "rudeness_name": row["rudeness_name"],
                        "category": row["category"],
                        "subcategory": row["subcategory"],
                    },
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
    metadata = [
        "example_id",
        "condition",
        "rudeness_type",
        "rudeness_name",
        "category",
        "subcategory",
    ]
    return ["example_id", "condition"], metadata, tasks


def _annotation_severity(row: dict[str, str]) -> int | None:
    for band in SEVERITY_BANDS:
        if row[f"is_abuse.{band}"] == "1":
            return band
    return None


def _severity_band(mean: float) -> int:
    # Nearest band; on an exact tie the more severe (smaller) band wins, so a
    # 50/50 annotator split is not diluted into the milder bucket.
    return min(SEVERITY_BANDS, key=lambda band: (abs(mean - band), band))


def _conversation_messages(
    prev_user: str, agent: str, user: str
) -> list[dict[str, str]]:
    # Gemma chat templates require strictly alternating turns starting with
    # the user, so context is included only when it forms a full
    # user -> assistant exchange before the labeled utterance.
    prev_user, agent, user = prev_user.strip(), agent.strip(), user.strip()
    if prev_user and agent:
        return [
            {"role": "user", "content": prev_user},
            {"role": "assistant", "content": agent},
            {"role": "user", "content": user},
        ]
    return [{"role": "user", "content": user}]


def load_convabuse(
    limit: int | None,
) -> tuple[list[str], list[str], list[Task]]:
    """Load ConvAbuse collapsed to one task per unique conversation snippet."""
    rows = _read_rows(CONVABUSE_FILE)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["conv_id"],
            row["prev_agent"],
            row["prev_user"],
            row["agent"],
            row["user"],
        )
        if row["user"].strip():
            groups[key].append(row)

    tasks: list[Task] = []
    for key, annotations in groups.items():
        severities = [
            severity
            for severity in (_annotation_severity(a) for a in annotations)
            if severity is not None
        ]
        if not severities:
            continue
        mean = sum(severities) / len(severities)
        abusive_votes = sum(1 for severity in severities if severity < 0)
        row: dict[str, object] = {
            "example_id": min(int(a["example_no"]) for a in annotations),
            "conv_id": annotations[0]["conv_id"],
            "bot": annotations[0]["bot"],
            "n_annotators": len(severities),
            "severity_mean": round(mean, 4),
            "severity_band": _severity_band(mean),
            "abusive_majority": abusive_votes > len(severities) / 2,
        }
        for flag in CONVABUSE_FLAGS:
            votes = [int(a[flag]) for a in annotations]
            row[flag.replace(".", "_") + "_frac"] = round(
                sum(votes) / len(votes), 4
            )
        tasks.append(
            {
                "row": row,
                "messages": _conversation_messages(
                    annotations[0]["prev_user"],
                    annotations[0]["agent"],
                    annotations[0]["user"],
                ),
            }
        )

    tasks.sort(key=lambda task: task["row"]["example_id"])  # type: ignore[index]
    if limit is not None:
        tasks = tasks[:limit]
    metadata = [
        "example_id",
        "conv_id",
        "bot",
        "n_annotators",
        "severity_mean",
        "severity_band",
        "abusive_majority",
    ] + [flag.replace(".", "_") + "_frac" for flag in CONVABUSE_FLAGS]
    return ["example_id"], metadata, tasks
