"""Deterministic normalized and analysis-ready Parquet exports.

Rudeness-label results produced from these files are associations, not causal
effects.  Opposition allocations are sign-inverted in the agreement exports so
positive action always means pro-continuation.  Voters are repeated stochastic
samples from one policy: no i.i.d.-voter claim is made, and the voter is the
clustering unit for downstream inference.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import hashlib
import json
import os
import shutil
import tempfile

from quadratic_voting.experiment.analysis import AnalysisInputs, analyze
from quadratic_voting.experiment.types import (
    ElicitationArm,
    ExportDataset,
    LikertRating,
    VotingRegime,
)


class ExportStore(Protocol):
    def export_rows(self, dataset: ExportDataset) -> tuple[dict[str, object], ...]: ...

    def candidate_rows(self) -> tuple[dict[str, object], ...]: ...

    def round_candidate_rows(self) -> tuple[dict[str, object], ...]: ...

    def source_annotation_rows(self) -> tuple[dict[str, object], ...]: ...

    def candidate_presentation_rows(self) -> tuple[dict[str, object], ...]: ...

    def candidate_turn_rows(self) -> tuple[dict[str, object], ...]: ...

    def voter_permutation_rows(self) -> tuple[dict[str, object], ...]: ...

    def experiment_config_rows(self) -> tuple[dict[str, object], ...]: ...


@dataclass(frozen=True, slots=True)
class ExportManifest:
    out_dir: Path
    files: tuple[Path, ...]


_S = pa.string()
_I = pa.int64()
_F = pa.float64()
_B = pa.bool_()
_SEED = pa.binary(8)


def _as_int(value: object) -> int:
    if isinstance(value, (str, int, float)):
        return int(value)
    raise TypeError(
        f"expected an integer-compatible exported value, received {value!r}"
    )


def _schema(fields: Sequence[tuple[str, pa.DataType]]) -> pa.Schema:
    sort_key = ",".join(name for name, _ in fields).encode("utf-8")
    return pa.schema(
        [pa.field(name, data_type) for name, data_type in fields],
        metadata={
            b"qv_schema_version": b"qv-analysis-export/v1",
            b"qv_sort_key": sort_key,
        },
    )


# Explicit schemas pin both empty exports and deterministic column order.  They
# intentionally mirror schema.sql rather than relying on pyarrow inference.
_NORMALIZED_SCHEMAS: Mapping[ExportDataset, pa.Schema] = {
    ExportDataset.RUNS: _schema(
        (
            ("run_id", _S),
            ("matched_set_id", _S),
            ("arm", _S),
            ("regime", _S),
            ("status", _S),
            ("pause_reason", _S),
            ("sample_id", _S),
            ("master_seed", _SEED),
            ("temperature", _F),
            ("top_p", _F),
            ("top_k", _I),
            ("max_new_tokens", _I),
            ("credit_budget", _I),
            ("attempt_limit", _I),
            ("voter_count", _I),
            ("max_consecutive_runtime_failures", _I),
            ("tie_policy", _S),
            ("presentation_policy", _S),
            ("action_format", _S),
            ("config_hash", _S),
        )
    ),
    ExportDataset.RUN_EXECUTIONS: _schema(
        (
            ("execution_id", _S),
            ("run_id", _S),
            ("python_version", _S),
            ("torch_version", _S),
            ("transformers_version", _S),
            ("uv_lock_hash", _S),
            ("device", _S),
            ("dtype", _S),
            ("hostname", _S),
            ("git_commit", _S),
            ("git_dirty", _B),
            ("started_at", _S),
            ("ended_at", _S),
            ("exit_reason", _S),
            ("drift_override", _B),
            ("environment_drift_json", _S),
            ("cuda_runtime_version", _S),
            ("nvidia_driver_version", _S),
            ("cudnn_version", _S),
            ("gpu_model", _S),
            ("gpu_count", _I),
            ("gpu_compute_capability", _S),
            ("gpu_uuid_hash", _S),
            ("os_name", _S),
            ("os_version", _S),
            ("kernel_version", _S),
            ("cpu_architecture", _S),
            ("deterministic_algorithms", _B),
            ("tf32_enabled", _B),
            ("cudnn_benchmark", _B),
            ("tracked_tree_hash", _S),
            ("binary_diff_sha256", _S),
            ("untracked_manifest_hash", _S),
            ("untracked_tree_hash", _S),
            ("hostname_hash", _S),
        )
    ),
    ExportDataset.VOTERS: _schema(
        (
            ("voter_id", _S),
            ("run_id", _S),
            ("voter_index", _I),
            ("permutation_seed", _SEED),
            ("permutation_algorithm", _S),
            ("permutation_coordinates_json", _S),
        )
    ),
    ExportDataset.ROUNDS: _schema(
        (("round_id", _S), ("run_id", _S), ("round_index", _I), ("phase", _S))
    ),
    ExportDataset.TURNS: _schema(
        (
            ("turn_id", _S),
            ("round_id", _S),
            ("voter_id", _S),
            ("kind", _S),
            ("status", _S),
        )
    ),
    ExportDataset.CALLS: _schema(
        (
            ("call_id", _S),
            ("turn_id", _S),
            ("attempt_index", _I),
            ("invocation_index", _I),
            ("status", _S),
            ("prompt_messages_json", _S),
            ("prompt_sha256", _S),
            ("seed", _SEED),
            ("raw_text", _S),
            ("prompt_token_count", _I),
            ("completion_token_count", _I),
            ("completion_token_ids_json", _S),
            ("stop_reason", _S),
            ("duration_ms", _I),
            ("diagnostics_json", _S),
            ("started_at", _S),
            ("committed_at", _S),
        )
    ),
    ExportDataset.VALIDATION_FAILURES: _schema(
        (
            ("failure_id", _S),
            ("call_id", _S),
            ("error_code", _S),
            ("ordinal", _I),
            ("message", _S),
        )
    ),
    ExportDataset.RUNTIME_FAILURES: _schema(
        (
            ("failure_id", _S),
            ("call_id", _S),
            ("kind", _S),
            ("diagnostics_json", _S),
            ("occurred_at", _S),
        )
    ),
    ExportDataset.BALLOTS: _schema(
        (
            ("ballot_id", _S),
            ("turn_id", _S),
            ("status", _S),
            ("accepted_call_id", _S),
            ("rationale", _S),
            ("engine_cost", _I),
        )
    ),
    ExportDataset.ALLOCATIONS: _schema(
        (("ballot_id", _S), ("candidate_id", _S), ("votes", _I))
    ),
    ExportDataset.STATEMENTS: _schema(
        (
            ("statement_id", _S),
            ("turn_id", _S),
            ("status", _S),
            ("accepted_call_id", _S),
        )
    ),
    ExportDataset.STATEMENT_ITEMS: _schema(
        (("statement_id", _S), ("candidate_id", _S), ("rating", _S), ("text", _S))
    ),
    ExportDataset.OUTCOMES: _schema(
        (
            ("round_id", _S),
            ("protected_candidate_id", _S),
            ("removed_candidate_id", _S),
            ("tie_flag", _B),
            ("sealed_at", _S),
        )
    ),
    ExportDataset.RNG_DRAWS: _schema(
        (
            ("draw_id", _S),
            ("run_id", _S),
            ("stream_domain", _S),
            ("round_index", _I),
            ("stream_name", _S),
            ("derived_seed", _SEED),
            ("seed_version", _S),
            ("coordinates_json", _S),
            ("algorithm_id", _S),
            ("draw_index", _I),
            ("chosen_candidate_id", _S),
        )
    ),
}

_RNG_DRAW_POPULATION_SCHEMA = _schema(
    (("draw_id", _S), ("position", _I), ("candidate_id", _S))
)

_PAIR_SCHEMA = _schema(
    (
        ("run_id", _S),
        ("arm", _S),
        ("regime", _S),
        ("round_index", _I),
        ("voter_index", _I),
        ("candidate_id", _S),
        ("rating_code", _I),
        ("signed_action", _I),
    )
)
_AGREEMENT_SCHEMA = _schema(
    (
        ("run_id", _S),
        ("arm", _S),
        ("regime", _S),
        ("round_index", _I),
        ("voter_index", _I),
        ("spearman_rho", _F),
        ("n_candidates", _I),
        ("missing_ballot", _B),
        ("missing_statement", _B),
        ("missing_ballot_count", _I),
        ("missing_statement_count", _I),
    )
)
_SURVIVAL_SCHEMA = _schema(
    (
        ("run_id", _S),
        ("arm", _S),
        ("regime", _S),
        ("candidate_id", _S),
        ("round_indices", pa.list_(_I)),
        ("votes_by_round", pa.list_(_I)),
        ("protected_rounds", pa.list_(_I)),
        ("removed_round", _I),
        ("survival_round", _I),
        ("winner", _B),
    )
)
_QUALITY_SCHEMA = _schema(
    (
        ("run_id", _S),
        ("arm", _S),
        ("regime", _S),
        ("invalid_attempts", _I),
        ("invalid_attempts_by_error_code", pa.map_(_S, _I)),
        ("correction_attempts", _I),
        ("retry_invocations", _I),
        ("abstentions", _I),
        ("invalid_missing_statements", _I),
        ("runtime_failures", _I),
        ("interruptions", _I),
        ("prompt_tokens", _I),
        ("completion_tokens", _I),
        ("total_duration_ms", _I),
    )
)
_TRAJECTORY_SCHEMA = _schema(
    (
        ("run_id", _S),
        ("arm", _S),
        ("regime", _S),
        ("round_index", _I),
        ("active_pool_size", _I),
        ("total_votes", _I),
        ("max_votes", _I),
        ("distinct_supported_candidates", _I),
    )
)

_CANDIDATE_METADATA_SCHEMA = _schema(
    (
        ("candidate_id", _S),
        ("release_id", _S),
        ("dataset_name", _S),
        ("release_version", _S),
        ("release_sha256", _S),
        ("source_row_id", _S),
        ("content_sha256", _S),
        ("rudeness_label", _S),
        ("label_policy_id", _S),
        ("label_policy_name", _S),
        ("label_policy_version", _S),
        ("label_policy_sha256", _S),
        ("presentation_id", _S),
        ("template_id", _S),
        ("presentation_template_name", _S),
        ("presentation_template_version", _S),
        ("presentation_template_sha256", _S),
        ("presentation_sha256", _S),
    )
)
_SOURCE_ANNOTATION_SCHEMA = _schema(
    (
        ("candidate_id", _S),
        ("release_id", _S),
        ("dataset_name", _S),
        ("release_version", _S),
        ("release_sha256", _S),
        ("annotation_index", _I),
        ("annotator_hash", _S),
        ("source_label", _S),
        ("source_value", _S),
    )
)
_CANDIDATE_PRESENTATION_SCHEMA = _schema(
    (
        ("presentation_id", _S),
        ("candidate_id", _S),
        ("release_id", _S),
        ("template_id", _S),
        ("template_name", _S),
        ("template_version", _S),
        ("body_sha256", _S),
        ("rendered_text", _S),
        ("rendered_sha256", _S),
    )
)
_CANDIDATE_SOURCE_TURN_SCHEMA = _schema(
    (
        ("candidate_id", _S),
        ("release_id", _S),
        ("dataset_name", _S),
        ("release_version", _S),
        ("release_sha256", _S),
        ("content_sha256", _S),
        ("turn_index", _I),
        ("role", _S),
        ("text", _S),
    )
)
_VOTER_PERMUTATION_SCHEMA = _schema(
    (
        ("run_id", _S),
        ("voter_id", _S),
        ("voter_index", _I),
        ("permutation_seed", pa.uint64()),
        ("permutation_algorithm", _S),
        ("permutation_coordinates_json", _S),
        ("position", _I),
        ("candidate_id", _S),
    )
)
_EXPERIMENT_CONFIG_SCHEMA = _schema(
    (
        ("config_id", _S),
        ("config_hash", _S),
        ("definition_hash", _S),
        ("sample_id", _S),
        ("master_seed", pa.uint64()),
        ("temperature", _F),
        ("top_p", _F),
        ("top_k", _I),
        ("max_new_tokens", _I),
        ("credit_budget", _I),
        ("ballot_max_corrections", _I),
        ("statement_max_corrections", _I),
        ("voter_count", _I),
        ("runtime_max_failures", _I),
        ("tie_policy", _S),
        ("presentation_policy", _S),
        ("action_format", _S),
        ("seed_version", _S),
        ("schema_version", _S),
        ("canonical_json_version", _S),
        ("prompt_encoding_version", _S),
        ("sampler_policy_version", _S),
        ("execution_class", _S),
        ("matched_set_id", _S),
        ("created_at", _S),
    )
)
_MODEL_DEFINITION_SCHEMA = _schema(
    (
        ("run_id", _S),
        ("model_id", _S),
        ("provider_id", _S),
        ("quantization_id", _S),
        ("artifact_repository", _S),
        ("artifact_revision", _S),
        ("presentation_template_id", _S),
        ("presentation_template_hash", _S),
        ("instruction_templates_json", _S),
        ("dataset_release_hash", _S),
        ("sample_artifact_hash", _S),
        ("runtime_id", _S),
        ("tokenizer_repository", _S),
        ("tokenizer_revision", _S),
        ("dtype", _S),
        ("route_registry_hash", _S),
        ("sampling_profile_hash", _S),
        ("instruction_profile_hash", _S),
        ("canonical_json_version", _S),
        ("prompt_encoding_version", _S),
        ("seed_version", _S),
        ("source_release_id", _S),
        ("label_policy_id", _S),
        ("label_policy_version", _S),
        ("label_policy_hash", _S),
        ("sample_id", _S),
    )
)
_ROUND_CANDIDATE_SCHEMA = _schema(
    (
        ("round_id", _S),
        ("run_id", _S),
        ("round_index", _I),
        ("candidate_id", _S),
        ("sample_position", _I),
    )
)
_CANDIDATE_ANALYSIS_SCHEMA = _schema(
    (
        ("matched_set_id", _S),
        ("run_id", _S),
        ("regime", _S),
        ("arm", _S),
        ("voter_index", _I),
        ("round_index", _I),
        ("candidate_id", _S),
        ("rating_code", _I),
        ("statement_text", _S),
        ("raw_votes", _I),
        ("signed_action", _I),
        ("rudeness_label", _S),
        ("label_policy_id", _S),
        ("label_policy_name", _S),
        ("label_policy_version", _S),
        ("label_policy_sha256", _S),
        ("source_annotations_json", _S),
        ("presentation_id", _S),
        ("presentation_template_id", _S),
        ("presentation_template_name", _S),
        ("presentation_template_version", _S),
        ("presentation_template_sha256", _S),
        ("presentation_sha256", _S),
        ("statement_status", _S),
        ("ballot_status", _S),
        ("missing_reason", _S),
        ("statement_retry_count", _I),
        ("ballot_retry_count", _I),
        ("statement_validation_failure_count", _I),
        ("ballot_validation_failure_count", _I),
        ("statement_runtime_failure_count", _I),
        ("ballot_runtime_failure_count", _I),
        ("sample_position", _I),
        ("active_pool_size", _I),
        ("intersection_pool_size", _I),
        ("in_all_run_intersection", _B),
        ("post_treatment_intersection", _B),
    )
)
_AGREEMENT_CELL_SCHEMA = _schema(
    (
        ("matched_set_id", _S),
        ("run_id", _S),
        ("arm", _S),
        ("regime", _S),
        ("round_index", _I),
        ("voter_index", _I),
        ("scope", _S),
        ("rudeness_label", _S),
        ("spearman_rho", _F),
        ("null_reason", _S),
        ("n_candidate_pairs", _I),
        ("n_eligible_candidates", _I),
        ("active_pool_size", _I),
        ("intersection_pool_size", _I),
        ("label_policy_version", _S),
        ("label_policy_id", _S),
        ("label_policy_sha256", _S),
    )
)
_AGREEMENT_SUMMARY_SCHEMA = _schema(
    (
        ("matched_set_id", _S),
        ("arm", _S),
        ("regime", _S),
        ("round_index", _I),
        ("scope", _S),
        ("rudeness_label", _S),
        ("mean_spearman_rho", _F),
        ("median_spearman_rho", _F),
        ("n_defined_cells", _I),
        ("n_total_eligible_cells", _I),
        ("n_candidate_pairs", _I),
        ("active_pool_size", _I),
        ("intersection_pool_size", _I),
        ("label_policy_version", _S),
        ("label_policy_id", _S),
        ("label_policy_sha256", _S),
        ("estimand_language", _S),
        ("n_null_missing_statement", _I),
        ("n_null_abstained_ballot", _I),
        ("n_null_n_lt_2", _I),
        ("n_null_constant_rating", _I),
        ("n_null_constant_action", _I),
    )
)
_CONTRAST_SCHEMA = _schema(
    (
        ("contrast_id", _S),
        ("matched_set_id", _S),
        ("regime", _S),
        ("round_index", _I),
        ("metric", _S),
        ("left_arm", _S),
        ("right_arm", _S),
        ("contrast_kind", _S),
        ("estimand_language", _S),
        ("claim_kind", _S),
        ("causal", _B),
        ("post_treatment", _B),
        ("original_pool_size", _I),
        ("intersection_pool_size", _I),
        ("estimate", _F),
        ("ci_lower", _F),
        ("ci_upper", _F),
        ("n_clusters", _I),
        ("bootstrap_replicates", _I),
        ("analysis_version", _S),
        ("seed_version", _S),
        ("bootstrap_version", _S),
        ("analysis_seed", pa.uint64()),
        ("voter_population", pa.list_(_I)),
        ("resample_sha256", _S),
        ("ci_method", _S),
        ("ci_level", _F),
    )
)
_BOOTSTRAP_SCHEMA = _schema(
    (
        ("contrast_id", _S),
        ("replicate_index", _I),
        ("estimate", _F),
        ("sampled_voter_indices", pa.list_(_I)),
    )
)

_RATING_CODES = {rating.value: index - 2 for index, rating in enumerate(LikertRating)}


def _rank(values: Sequence[int]) -> list[float]:
    """Return one-based mid-ranks, averaging rank positions for ties."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        midrank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            result[ordered[position][0]] = midrank
        start = end
    return result


def _spearman(left: Sequence[int], right: Sequence[int]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank, strict=True)
    )
    denominator = (
        sum((value - left_mean) ** 2 for value in left_rank)
        * sum((value - right_mean) ** 2 for value in right_rank)
    ) ** 0.5
    return None if denominator == 0 else numerator / denominator


def _table(rows: Sequence[Mapping[str, object]], schema: pa.Schema) -> pa.Table:
    normalized = []
    for row in rows:
        values: dict[str, object] = {}
        for field in schema:
            value = row.get(field.name)
            values[field.name] = (
                bool(value)
                if value is not None and pa.types.is_boolean(field.type)
                else value
            )
        normalized.append(values)
    return pa.Table.from_pylist(normalized, schema=schema)


def _sort_rows(
    rows: Sequence[Mapping[str, object]], schema: pa.Schema
) -> list[Mapping[str, object]]:
    """Sort rows by the complete versioned schema without mixed-type comparisons."""

    def key(row: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(
            "" if row.get(field.name) is None else repr(row.get(field.name))
            for field in schema
        )

    return sorted(rows, key=key)


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, object]], schema: pa.Schema
) -> None:
    pq.write_table(
        _table(_sort_rows(rows, schema), schema),
        path,
        compression="zstd",
        version="2.6",
        write_statistics=True,
    )


def _required_accessor_rows(
    store: object, accessor_name: str
) -> tuple[dict[str, object], ...]:
    accessor = getattr(store, accessor_name, None)
    if not callable(accessor):
        raise RuntimeError(
            f"Parquet export failed because the durable store does not expose "
            f"ExportStore.{accessor_name}(). Provenance collection stopped in "
            "experiment.export._required_accessor_rows before any incomplete provenance "
            "dataset was written, so analysts will not receive silently nullable lineage. "
            f"Implement the read-only normalized {accessor_name} accessor in the core store "
            "and rerun export."
        )
    rows = accessor()
    if not isinstance(rows, tuple) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(
            f"Parquet export failed because ExportStore.{accessor_name}() returned "
            f"{type(rows).__name__}, not tuple[dict[str, object], ...]. Validation failed "
            "at the export boundary before writing an ambiguous schema. Return normalized "
            "mapping rows in deterministic order and retry."
        )
    return rows


def _filter_for_matched_set(
    data: dict[ExportDataset, tuple[dict[str, object], ...]],
    matched_set_id: str | None,
) -> tuple[str, dict[ExportDataset, tuple[dict[str, object], ...]]]:
    available = sorted({str(row["matched_set_id"]) for row in data[ExportDataset.RUNS]})
    if matched_set_id is None:
        if not available:
            matched_set_id = "__empty__"
            return matched_set_id, data
        if len(available) != 1:
            raise ValueError(
                "Parquet export requires an explicit matched_set_id when the store contains "
                f"{len(available)} matched sets ({available}). Filtering failed in "
                "experiment.export._filter_for_matched_set before staging, so no cross-set "
                "rows were exported. Pass the requested --matched-set value to export_parquet."
            )
        matched_set_id = available[0]
    if matched_set_id not in available:
        raise ValueError(
            f"Parquet export cannot find matched set {matched_set_id!r}; available IDs are "
            f"{available}. Selection failed before staging, so no output was published. "
            "Use an existing matched-set ID from inspect and retry."
        )
    runs = tuple(
        row
        for row in data[ExportDataset.RUNS]
        if str(row["matched_set_id"]) == matched_set_id
    )
    run_ids = {str(row["run_id"]) for row in runs}
    voters = tuple(
        row for row in data[ExportDataset.VOTERS] if str(row["run_id"]) in run_ids
    )
    voter_ids = {str(row["voter_id"]) for row in voters}
    rounds = tuple(
        row for row in data[ExportDataset.ROUNDS] if str(row["run_id"]) in run_ids
    )
    round_ids = {str(row["round_id"]) for row in rounds}
    turns = tuple(
        row
        for row in data[ExportDataset.TURNS]
        if str(row["round_id"]) in round_ids and str(row["voter_id"]) in voter_ids
    )
    turn_ids = {str(row["turn_id"]) for row in turns}
    calls = tuple(
        row for row in data[ExportDataset.CALLS] if str(row["turn_id"]) in turn_ids
    )
    call_ids = {str(row["call_id"]) for row in calls}
    ballots = tuple(
        row for row in data[ExportDataset.BALLOTS] if str(row["turn_id"]) in turn_ids
    )
    ballot_ids = {str(row["ballot_id"]) for row in ballots}
    statements = tuple(
        row for row in data[ExportDataset.STATEMENTS] if str(row["turn_id"]) in turn_ids
    )
    statement_ids = {str(row["statement_id"]) for row in statements}
    filtered = dict(data)
    filtered.update(
        {
            ExportDataset.RUNS: runs,
            ExportDataset.RUN_EXECUTIONS: tuple(
                row
                for row in data[ExportDataset.RUN_EXECUTIONS]
                if str(row["run_id"]) in run_ids
            ),
            ExportDataset.VOTERS: voters,
            ExportDataset.ROUNDS: rounds,
            ExportDataset.TURNS: turns,
            ExportDataset.CALLS: calls,
            ExportDataset.VALIDATION_FAILURES: tuple(
                row
                for row in data[ExportDataset.VALIDATION_FAILURES]
                if str(row["call_id"]) in call_ids
            ),
            ExportDataset.RUNTIME_FAILURES: tuple(
                row
                for row in data[ExportDataset.RUNTIME_FAILURES]
                if str(row["call_id"]) in call_ids
            ),
            ExportDataset.BALLOTS: ballots,
            ExportDataset.ALLOCATIONS: tuple(
                row
                for row in data[ExportDataset.ALLOCATIONS]
                if str(row["ballot_id"]) in ballot_ids
            ),
            ExportDataset.STATEMENTS: statements,
            ExportDataset.STATEMENT_ITEMS: tuple(
                row
                for row in data[ExportDataset.STATEMENT_ITEMS]
                if str(row["statement_id"]) in statement_ids
            ),
            ExportDataset.OUTCOMES: tuple(
                row
                for row in data[ExportDataset.OUTCOMES]
                if str(row["round_id"]) in round_ids
            ),
            ExportDataset.RNG_DRAWS: tuple(
                row
                for row in data[ExportDataset.RNG_DRAWS]
                if str(row["run_id"]) in run_ids
            ),
        }
    )
    return matched_set_id, filtered


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Artifact publication refused to replace existing directory {destination}. "
            "Atomic publication stopped in experiment.export._publish_directory while the "
            "complete new artifact remained staged; choose a new output directory or remove "
            "the old complete artifact explicitly, then retry."
        )
    os.replace(staging, destination)
    _fsync_directory(destination.parent)


def _relations(
    data: Mapping[ExportDataset, tuple[dict[str, object], ...]],
) -> dict[str, object]:
    rounds = {str(row["round_id"]): row for row in data[ExportDataset.ROUNDS]}
    voters = {str(row["voter_id"]): row for row in data[ExportDataset.VOTERS]}
    turns = {str(row["turn_id"]): row for row in data[ExportDataset.TURNS]}
    ballots = {str(row["turn_id"]): row for row in data[ExportDataset.BALLOTS]}
    statements = {str(row["turn_id"]): row for row in data[ExportDataset.STATEMENTS]}
    allocations: dict[str, dict[str, int]] = defaultdict(dict)
    for row in data[ExportDataset.ALLOCATIONS]:
        allocations[str(row["ballot_id"])][str(row["candidate_id"])] = _as_int(
            row["votes"]
        )
    items: dict[str, dict[str, str]] = defaultdict(dict)
    for row in data[ExportDataset.STATEMENT_ITEMS]:
        items[str(row["statement_id"])][str(row["candidate_id"])] = str(row["rating"])
    return {
        "rounds": rounds,
        "voters": voters,
        "turns": turns,
        "ballots": ballots,
        "statements": statements,
        "allocations": allocations,
        "items": items,
    }


def _candidate_universes(
    data: Mapping[ExportDataset, tuple[dict[str, object], ...]],
    relations: Mapping[str, object],
) -> dict[str, set[str]]:
    """Recover matched-set candidate universes from normalized observations."""
    rounds = relations["rounds"]
    turns = relations["turns"]
    ballots = relations["ballots"]
    allocations = relations["allocations"]
    items = relations["items"]
    assert isinstance(rounds, dict) and isinstance(turns, dict)
    assert (
        isinstance(ballots, dict)
        and isinstance(allocations, dict)
        and isinstance(items, dict)
    )
    runs = {str(row["run_id"]): row for row in data[ExportDataset.RUNS]}
    observed: dict[str, set[str]] = defaultdict(set)
    for ballot in data[ExportDataset.BALLOTS]:
        turn = turns[str(ballot["turn_id"])]
        run_id = str(rounds[str(turn["round_id"])]["run_id"])
        observed[run_id].update(allocations.get(str(ballot["ballot_id"]), {}))
    for statement in data[ExportDataset.STATEMENTS]:
        turn = turns[str(statement["turn_id"])]
        run_id = str(rounds[str(turn["round_id"])]["run_id"])
        observed[run_id].update(items.get(str(statement["statement_id"]), {}))
    for outcome in data[ExportDataset.OUTCOMES]:
        run_id = str(rounds[str(outcome["round_id"])]["run_id"])
        observed[run_id].add(str(outcome["removed_candidate_id"]))
        if outcome["protected_candidate_id"] is not None:
            observed[run_id].add(str(outcome["protected_candidate_id"]))
    matched: dict[str, set[str]] = defaultdict(set)
    for run_id, candidates in observed.items():
        matched[str(runs[run_id]["matched_set_id"])].update(candidates)
    return {
        run_id: set(matched[str(run["matched_set_id"])]) for run_id, run in runs.items()
    }


def _agreement_rows(
    data: Mapping[ExportDataset, tuple[dict[str, object], ...]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rel = _relations(data)
    rounds = rel["rounds"]
    voters = rel["voters"]
    turns = rel["turns"]
    ballots = rel["ballots"]
    statements = rel["statements"]
    allocations = rel["allocations"]
    items = rel["items"]
    assert (
        isinstance(rounds, dict)
        and isinstance(voters, dict)
        and isinstance(turns, dict)
    )
    assert isinstance(ballots, dict) and isinstance(statements, dict)
    assert isinstance(allocations, dict) and isinstance(items, dict)
    runs = {str(row["run_id"]): row for row in data[ExportDataset.RUNS]}
    universes = _candidate_universes(data, rel)
    turn_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for turn in data[ExportDataset.TURNS]:
        turn_by_key[
            (str(turn["round_id"]), str(turn["voter_id"]) + ":" + str(turn["kind"]))
        ] = turn
    active_by_round: dict[str, set[str]] = defaultdict(set)
    for source_statement in data[ExportDataset.STATEMENTS]:
        round_turn = turns.get(str(source_statement["turn_id"]))
        if round_turn is not None:
            active_by_round[str(round_turn["round_id"])].update(
                items.get(str(source_statement["statement_id"]), {})
            )

    summaries: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    missing_by_round: Counter[tuple[str, int, str]] = Counter()
    for round_row in data[ExportDataset.ROUNDS]:
        run_id = str(round_row["run_id"])
        run = runs[run_id]
        if str(run["arm"]) == ElicitationArm.ACTION_ONLY.value:
            continue
        round_id = str(round_row["round_id"])
        run_voters = sorted(
            (row for row in data[ExportDataset.VOTERS] if str(row["run_id"]) == run_id),
            key=lambda row: _as_int(row["voter_index"]),
        )
        for voter in run_voters:
            voter_id = str(voter["voter_id"])
            ballot_turn = turn_by_key.get((round_id, voter_id + ":ballot"))
            statement_turn = turn_by_key.get((round_id, voter_id + ":statement"))
            ballot = (
                None
                if ballot_turn is None
                else ballots.get(str(ballot_turn["turn_id"]))
            )
            statement = (
                None
                if statement_turn is None
                else statements.get(str(statement_turn["turn_id"]))
            )
            # Pending turns are not missing observations. Missingness is recorded
            # only after both required terminal records cross the round barrier.
            if ballot is None or statement is None:
                continue
            missing_ballot = ballot is None or str(ballot["status"]) != "accepted"
            missing_statement = (
                statement is None or str(statement["status"]) != "accepted"
            )
            if missing_ballot:
                missing_by_round[
                    (run_id, _as_int(round_row["round_index"]), "ballot")
                ] += 1
            if missing_statement:
                missing_by_round[
                    (run_id, _as_int(round_row["round_index"]), "statement")
                ] += 1
            ratings = (
                {}
                if statement is None
                else items.get(str(statement["statement_id"]), {})
            )
            votes = (
                {} if ballot is None else allocations.get(str(ballot["ballot_id"]), {})
            )
            candidate_ids = sorted(active_by_round[round_id])
            if not candidate_ids:
                removed_before = {
                    str(outcome["removed_candidate_id"])
                    for outcome in data[ExportDataset.OUTCOMES]
                    if str(rounds[str(outcome["round_id"])]["run_id"]) == run_id
                    and _as_int(rounds[str(outcome["round_id"])]["round_index"])
                    < _as_int(round_row["round_index"])
                }
                candidate_ids = sorted(universes[run_id] - removed_before)
            row = {
                "run_id": run_id,
                "arm": run["arm"],
                "regime": run["regime"],
                "round_index": _as_int(round_row["round_index"]),
                "voter_index": _as_int(voter["voter_index"]),
                "spearman_rho": None,
                "n_candidates": len(candidate_ids),
                "missing_ballot": missing_ballot,
                "missing_statement": missing_statement,
            }
            if not missing_ballot and not missing_statement:
                rating_codes = [
                    _RATING_CODES[ratings[candidate]] for candidate in candidate_ids
                ]
                sign = 1 if str(run["regime"]) == VotingRegime.SUPPORT.value else -1
                signed_actions = [
                    sign * _as_int(votes.get(candidate, 0))
                    for candidate in candidate_ids
                ]
                row["spearman_rho"] = _spearman(rating_codes, signed_actions)
                for candidate, rating_code, signed_action in zip(
                    candidate_ids, rating_codes, signed_actions, strict=True
                ):
                    pairs.append(
                        {
                            **{
                                key: row[key]
                                for key in (
                                    "run_id",
                                    "arm",
                                    "regime",
                                    "round_index",
                                    "voter_index",
                                )
                            },
                            "candidate_id": candidate,
                            "rating_code": rating_code,
                            "signed_action": signed_action,
                        }
                    )
            pending.append(row)
    for row in pending:
        key = (str(row["run_id"]), _as_int(row["round_index"]))
        row["missing_ballot_count"] = missing_by_round[(*key, "ballot")]
        row["missing_statement_count"] = missing_by_round[(*key, "statement")]
        summaries.append(row)
    return pairs, summaries


def _candidate_and_trajectory_rows(
    data: Mapping[ExportDataset, tuple[dict[str, object], ...]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rel = _relations(data)
    rounds = rel["rounds"]
    turns = rel["turns"]
    ballots = rel["ballots"]
    allocations = rel["allocations"]
    items = rel["items"]
    assert isinstance(rounds, dict) and isinstance(turns, dict)
    assert (
        isinstance(ballots, dict)
        and isinstance(allocations, dict)
        and isinstance(items, dict)
    )
    runs = {str(row["run_id"]): row for row in data[ExportDataset.RUNS]}
    universes = _candidate_universes(data, rel)
    totals: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    candidates_by_run: dict[str, set[str]] = defaultdict(set)
    for ballot in data[ExportDataset.BALLOTS]:
        turn = turns[str(ballot["turn_id"])]
        round_row = rounds[str(turn["round_id"])]
        run_id, round_index = (
            str(round_row["run_id"]),
            _as_int(round_row["round_index"]),
        )
        for candidate, votes in allocations.get(str(ballot["ballot_id"]), {}).items():
            totals[(run_id, round_index)][candidate] += votes
            candidates_by_run[run_id].add(candidate)
    for statement in data[ExportDataset.STATEMENTS]:
        turn = turns[str(statement["turn_id"])]
        run_id = str(rounds[str(turn["round_id"])]["run_id"])
        candidates_by_run[run_id].update(items.get(str(statement["statement_id"]), {}))
    outcomes: dict[tuple[str, int], dict[str, object]] = {}
    for outcome in data[ExportDataset.OUTCOMES]:
        round_row = rounds[str(outcome["round_id"])]
        key = (str(round_row["run_id"]), _as_int(round_row["round_index"]))
        outcomes[key] = outcome
        candidates_by_run[key[0]].add(str(outcome["removed_candidate_id"]))
        if outcome["protected_candidate_id"] is not None:
            candidates_by_run[key[0]].add(str(outcome["protected_candidate_id"]))
    survival: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    for run_id, run in sorted(runs.items()):
        run_rounds = sorted(
            _as_int(row["round_index"])
            for row in data[ExportDataset.ROUNDS]
            if str(row["run_id"]) == run_id
        )
        sealed_rounds = sorted(index for rid, index in outcomes if rid == run_id)
        removed = {
            str(outcomes[(run_id, index)]["removed_candidate_id"]): index
            for index in sealed_rounds
        }
        candidates = sorted(universes[run_id] | candidates_by_run[run_id])
        for index in run_rounds:
            active = sum(
                1 for candidate in candidates if removed.get(candidate, index) >= index
            )
            values = totals[(run_id, index)]
            trajectories.append(
                {
                    "run_id": run_id,
                    "arm": run["arm"],
                    "regime": run["regime"],
                    "round_index": index,
                    "active_pool_size": active,
                    "total_votes": sum(values.values()),
                    "max_votes": max(values.values(), default=0),
                    "distinct_supported_candidates": sum(
                        votes > 0 for votes in values.values()
                    ),
                }
            )
        final_round = max(sealed_rounds, default=max(run_rounds, default=0))
        for candidate in candidates:
            removed_round = removed.get(candidate)
            survival.append(
                {
                    "run_id": run_id,
                    "arm": run["arm"],
                    "regime": run["regime"],
                    "candidate_id": candidate,
                    "round_indices": run_rounds,
                    "votes_by_round": [
                        totals[(run_id, index)][candidate] for index in run_rounds
                    ],
                    "protected_rounds": [
                        index
                        for index in sealed_rounds
                        if outcomes[(run_id, index)]["protected_candidate_id"]
                        == candidate
                    ],
                    "removed_round": removed_round,
                    "survival_round": final_round
                    if removed_round is None
                    else removed_round,
                    "winner": bool(
                        run["status"] == "complete" and removed_round is None
                    ),
                }
            )
    return survival, trajectories


def _quality_rows(
    data: Mapping[ExportDataset, tuple[dict[str, object], ...]],
) -> list[dict[str, object]]:
    rel = _relations(data)
    turns = rel["turns"]
    rounds = rel["rounds"]
    assert isinstance(turns, dict) and isinstance(rounds, dict)
    call_run: dict[str, str] = {}
    calls_by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for call in data[ExportDataset.CALLS]:
        turn = turns[str(call["turn_id"])]
        run_id = str(rounds[str(turn["round_id"])]["run_id"])
        call_run[str(call["call_id"])] = run_id
        calls_by_run[run_id].append(call)
    errors: dict[str, Counter[str]] = defaultdict(Counter)
    for failure in data[ExportDataset.VALIDATION_FAILURES]:
        errors[call_run[str(failure["call_id"])]][str(failure["error_code"])] += 1
    runtime = Counter(
        call_run[str(row["call_id"])] for row in data[ExportDataset.RUNTIME_FAILURES]
    )
    ballot_run = {
        str(row["turn_id"]): str(
            rounds[str(turns[str(row["turn_id"])]["round_id"])]["run_id"]
        )
        for row in data[ExportDataset.BALLOTS]
    }
    statement_run = {
        str(row["turn_id"]): str(
            rounds[str(turns[str(row["turn_id"])]["round_id"])]["run_id"]
        )
        for row in data[ExportDataset.STATEMENTS]
    }
    abstentions = Counter(
        ballot_run[str(row["turn_id"])]
        for row in data[ExportDataset.BALLOTS]
        if row["status"] == "abstained"
    )
    missing = Counter(
        statement_run[str(row["turn_id"])]
        for row in data[ExportDataset.STATEMENTS]
        if row["status"] == "invalid-missing"
    )
    runs = {str(row["run_id"]): row for row in data[ExportDataset.RUNS]}
    result = []
    for run_id, run in sorted(runs.items()):
        calls = calls_by_run[run_id]
        error_items = sorted(errors[run_id].items())
        result.append(
            {
                "run_id": run_id,
                "arm": run["arm"],
                "regime": run["regime"],
                "invalid_attempts": sum(
                    1
                    for call in calls
                    if str(call["call_id"])
                    in {
                        str(failure["call_id"])
                        for failure in data[ExportDataset.VALIDATION_FAILURES]
                    }
                ),
                "invalid_attempts_by_error_code": error_items,
                "correction_attempts": sum(
                    _as_int(call["attempt_index"]) > 0 for call in calls
                ),
                "retry_invocations": sum(
                    _as_int(call["invocation_index"]) > 0 for call in calls
                ),
                "abstentions": abstentions[run_id],
                "invalid_missing_statements": missing[run_id],
                "runtime_failures": runtime[run_id],
                "interruptions": sum(call["status"] == "interrupted" for call in calls),
                "prompt_tokens": sum(
                    _as_int(call["prompt_token_count"] or 0) for call in calls
                ),
                "completion_tokens": sum(
                    _as_int(call["completion_token_count"] or 0) for call in calls
                ),
                "total_duration_ms": sum(
                    _as_int(call["duration_ms"] or 0) for call in calls
                ),
            }
        )
    return result


def export_parquet(
    store: ExportStore, out_dir: Path, *, matched_set_id: str | None = None
) -> ExportManifest:
    """Export normalized and summary data with schemas pinned for empty row sets.

    The explicit per-dataset ``pyarrow.Schema`` constants above ensure that an
    empty database still emits readable files with stable columns and types.
    """
    data = {dataset: store.export_rows(dataset) for dataset in ExportDataset}
    matched_set_id, data = _filter_for_matched_set(data, matched_set_id)
    run_ids = {str(row["run_id"]) for row in data[ExportDataset.RUNS]}
    candidates = store.candidate_rows()
    round_candidates = tuple(
        row for row in store.round_candidate_rows() if str(row["run_id"]) in run_ids
    )
    candidate_ids = {str(row["candidate_id"]) for row in round_candidates}
    source_annotations = tuple(
        row
        for row in store.source_annotation_rows()
        if str(row["candidate_id"]) in candidate_ids
    )
    candidate_presentations = tuple(
        row
        for row in store.candidate_presentation_rows()
        if str(row["candidate_id"]) in candidate_ids
    )
    candidate_source_turns = tuple(
        row
        for row in _required_accessor_rows(store, "candidate_turn_rows")
        if str(row["candidate_id"]) in candidate_ids
    )
    candidates = tuple(
        row for row in candidates if str(row["candidate_id"]) in candidate_ids
    )
    voter_permutations = tuple(
        row for row in store.voter_permutation_rows() if str(row["run_id"]) in run_ids
    )
    experiment_configs = tuple(
        row
        for row in store.experiment_config_rows()
        if str(row["matched_set_id"]) == matched_set_id
    )
    model_definitions = tuple(
        row
        for row in _required_accessor_rows(store, "run_definition_rows")
        if str(row["run_id"]) in run_ids
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent)
    )
    files: list[Path] = []
    try:
        for dataset in ExportDataset:
            path = staging / f"{dataset.value}.parquet"
            _write_parquet(path, data[dataset], _NORMALIZED_SCHEMAS[dataset])
            files.append(path)
        rng_populations: list[dict[str, object]] = []
        for row in data[ExportDataset.RNG_DRAWS]:
            population = row.get("population")
            if not isinstance(population, list):
                raise TypeError(
                    f"RNG draw {row['draw_id']} has no ordered population list at the export "
                    "boundary. Normalized RNG publication stopped before staging an incomplete "
                    "draw. Ensure export_rows(RNG_DRAWS) includes its persisted population and retry."
                )
            rng_populations.extend(
                {
                    "draw_id": row["draw_id"],
                    "position": position,
                    "candidate_id": str(candidate),
                }
                for position, candidate in enumerate(population)
            )
        supplemental = (
            ("rng_draw_populations", rng_populations, _RNG_DRAW_POPULATION_SCHEMA),
            ("candidate_metadata", candidates, _CANDIDATE_METADATA_SCHEMA),
            ("source_annotations", source_annotations, _SOURCE_ANNOTATION_SCHEMA),
            (
                "candidate_presentations",
                candidate_presentations,
                _CANDIDATE_PRESENTATION_SCHEMA,
            ),
            (
                "candidate_source_turns",
                candidate_source_turns,
                _CANDIDATE_SOURCE_TURN_SCHEMA,
            ),
            ("voter_permutations", voter_permutations, _VOTER_PERMUTATION_SCHEMA),
            (
                "experiment_configurations",
                experiment_configs,
                _EXPERIMENT_CONFIG_SCHEMA,
            ),
            ("model_definitions", model_definitions, _MODEL_DEFINITION_SCHEMA),
            ("round_candidates", round_candidates, _ROUND_CANDIDATE_SCHEMA),
        )
        for name, rows, schema in supplemental:
            path = staging / f"{name}.parquet"
            _write_parquet(path, rows, schema)
            files.append(path)

        analysis = analyze(
            AnalysisInputs(
                runs=data[ExportDataset.RUNS],
                voters=data[ExportDataset.VOTERS],
                rounds=data[ExportDataset.ROUNDS],
                round_candidate_rows=round_candidates,
                candidate_rows=candidates,
                source_annotation_rows=source_annotations,
                run_definition_rows=model_definitions,
                turns=data[ExportDataset.TURNS],
                calls=data[ExportDataset.CALLS],
                validation_failures=data[ExportDataset.VALIDATION_FAILURES],
                runtime_failures=data[ExportDataset.RUNTIME_FAILURES],
                ballots=data[ExportDataset.BALLOTS],
                allocations=data[ExportDataset.ALLOCATIONS],
                statements=data[ExportDataset.STATEMENTS],
                statement_items=data[ExportDataset.STATEMENT_ITEMS],
                outcomes=data[ExportDataset.OUTCOMES],
            )
        )
        # Keep the original pair export name as a compatibility view while adding
        # the complete candidate-level relation required by analysis/v1.
        pairs = [
            {
                "run_id": row["run_id"],
                "arm": row["arm"],
                "regime": row["regime"],
                "round_index": row["round_index"],
                "voter_index": row["voter_index"],
                "candidate_id": row["candidate_id"],
                "rating_code": row["rating_code"],
                "signed_action": row["signed_action"],
            }
            for row in analysis.candidate_rows
            if row["rating_code"] is not None and row["signed_action"] is not None
        ]
        summaries: Sequence[tuple[str, Sequence[Mapping[str, object]], pa.Schema]] = (
            ("preference_action_pairs", pairs, _PAIR_SCHEMA),
            ("candidate_analysis", analysis.candidate_rows, _CANDIDATE_ANALYSIS_SCHEMA),
            (
                "preference_action_agreement",
                analysis.agreement_cells,
                _AGREEMENT_CELL_SCHEMA,
            ),
            (
                "preference_action_summary",
                analysis.agreement_summaries,
                _AGREEMENT_SUMMARY_SCHEMA,
            ),
            ("paired_contrasts", analysis.contrasts, _CONTRAST_SCHEMA),
            ("bootstrap_replicates", analysis.bootstrap_replicates, _BOOTSTRAP_SCHEMA),
            ("candidate_survival", analysis.candidate_survival, _SURVIVAL_SCHEMA),
            ("run_quality", analysis.run_quality, _QUALITY_SCHEMA),
            ("round_trajectories", analysis.round_trajectories, _TRAJECTORY_SCHEMA),
        )
        for name, analysis_rows, schema in summaries:
            path = staging / f"{name}.parquet"
            _write_parquet(path, analysis_rows, schema)
            files.append(path)
        manifest_payload = {
            "version": "qv-export-manifest/v1",
            "matched_set_id": matched_set_id,
            "files": [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(files, key=lambda path: path.name)
            ],
        }
        manifest_path = staging / "export-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        files.append(manifest_path)
        for path in files:
            _fsync_file(path)
        _fsync_directory(staging)
        _publish_directory(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    published = tuple(out_dir / path.name for path in files)
    return ExportManifest(out_dir, published)
