"""One-command orchestration for the reviewed default six-run pilot."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import secrets
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from huggingface_hub.constants import HF_HUB_CACHE

from llm_runtime import (
    LocalTransformersRoute,
    ModelId,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from llm_runtime.transformers import (
    Device,
    create_transformers_runtime,
    download_transformers_artifact,
)
from quadratic_voting.experiment import gemma
from quadratic_voting.experiment.catalog import (
    DEFAULT_PRESENTATION_TEMPLATE_BODY,
    DEFAULT_PRESENTATION_TEMPLATE_NAME,
    DEFAULT_PRESENTATION_TEMPLATE_VERSION,
    render_release_candidate_cards,
)
from quadratic_voting.experiment.config import MatchedSetConfigV1
from quadratic_voting.experiment.sample_file import (
    replace_and_fsync_directory,
    write_fsynced_temp,
)
from quadratic_voting.experiment.store import (
    acquire_writer_lock,
    open_sqlite_store,
    seed_to_blob,
)
from quadratic_voting.experiment.transcript import (
    INSTRUCTION_TEMPLATE_VERSION,
    TEMPLATE_BODIES,
)
from quadratic_voting.experiment.types import (
    ElicitationArm,
    ReleaseId,
    SamplingProfile,
    TemplateId,
    VoterGenerator,
    VotingRegime,
)

_PIPELINE_VERSION: Final[str] = "qv-default-pipeline/v1"
_REVIEW_VERSION: Final[str] = "default-pilot-review/v2"
_LABEL_POLICY_VERSION: Final[str] = "majority-severity-negative-complete-context/v3"
_DEFAULT_SEED: Final[int] = 20260815
_MODEL_SNAPSHOT_FILES: Final[frozenset[str]] = frozenset(
    {
        ".gitattributes",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
)


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be greater than zero")
    return parsed


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error
    if not 0 <= parsed <= (1 << 64) - 1:
        raise argparse.ArgumentTypeError(f"{value!r} is outside the uint64 seed range")
    return parsed


def _invoke(args: argparse.Namespace, command: list[str]) -> str:
    from quadratic_voting.experiment.cli.main import main

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = main(
            ["--db", str(args.db), *command],
            generator_factory=args.generator_factory,
        )
    text = output.getvalue()
    if text:
        print(text, end="", flush=True)
    if status != 0:
        raise RuntimeError(
            f"Default pipeline stopped because `{' '.join(command)}` returned status "
            f"{status}. Completed database work remains resumable; rerun the same pipeline "
            "command after correcting the reported failure."
        )
    return text


def _default_generator(args: argparse.Namespace) -> VoterGenerator:
    """Construct the reviewed pipeline's one local Gemma generator."""
    route = resolve_route(ModelId.GEMMA_4_E2B_IT, ProviderId.LOCAL, QuantizationId.BF16)
    if not isinstance(route, LocalTransformersRoute):
        raise AssertionError("the reviewed default Gemma route must be local")
    runtime = create_transformers_runtime(
        route, cache_dir=args.cache_dir, device=args.device
    )
    return gemma.GemmaVoterGenerator(runtime)


def _shared_default_generator_factory(
    args: argparse.Namespace,
) -> Callable[[SamplingProfile], VoterGenerator]:
    """Lazily share the default runtime across incomplete nested ``run`` commands."""
    generator: VoterGenerator | None = None

    def factory(_profile: SamplingProfile) -> VoterGenerator:
        nonlocal generator
        if generator is None:
            generator = _default_generator(args)
        return generator

    return factory


def _value(text: str, key: str) -> str:
    marker = f"{key}="
    for line in text.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].split()[0]
    raise RuntimeError(
        f"Default pipeline could not parse {key!r} from a successful command response. "
        "The lower-level CLI output contract changed; update pipeline_cmds before retrying."
    )


def _write_json(path: Path, value: object) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temp = write_fsynced_temp(path, content)
    replace_and_fsync_directory(temp, path)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _database_identity(database: Path, *, create: bool) -> str | None:
    connection = sqlite3.connect(database)
    try:
        try:
            row = connection.execute(
                "SELECT database_id FROM pipeline_database_identity"
            ).fetchone()
        except sqlite3.OperationalError as error:
            if create:
                raise RuntimeError(
                    f"Pipeline database identity table is missing in {database}; migrate "
                    "the database with the current experiment schema before retrying."
                ) from error
            return None
        if row is not None:
            return str(row[0])
        if not create:
            return None
        identity = secrets.token_hex(16)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO pipeline_database_identity(database_id) VALUES (?)",
            (identity,),
        )
        connection.commit()
        return identity
    finally:
        connection.close()


def _approve_label_policy(database: Path, review_sha256: str, *, command: str) -> None:
    with acquire_writer_lock(database, command=command) as writer_lock:
        with open_sqlite_store(
            database, writer_lock=writer_lock, require_writer_lock=True
        ) as store:
            row = store.connection.execute(
                "SELECT label_policy_id,reviewed,review_version,review_sha256 "
                "FROM label_policy WHERE name=? AND version=?",
                ("convabuse-rudeness", _LABEL_POLICY_VERSION),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "Default pipeline review binding failed because the ingested "
                    f"convabuse-rudeness/{_LABEL_POLICY_VERSION} policy is absent. "
                    "No review metadata changed; inspect catalog ingestion and retry."
                )
            expected = (1, _REVIEW_VERSION, review_sha256)
            actual = (int(row["reviewed"]), row["review_version"], row["review_sha256"])
            if actual == expected:
                return
            if actual != (0, None, None):
                raise RuntimeError(
                    "Default pipeline review binding refused existing, different review "
                    f"metadata {actual!r}; expected {expected!r}. Use a new database or an "
                    "explicit lower-level configuration rather than overwriting approval."
                )
            store.connection.execute("BEGIN IMMEDIATE")
            store.connection.execute(
                "UPDATE label_policy SET reviewed=1,review_version=?,review_sha256=? "
                "WHERE label_policy_id=?",
                (_REVIEW_VERSION, review_sha256, row["label_policy_id"]),
            )
            store.connection.commit()


def _build_config(
    args: argparse.Namespace,
    *,
    sample_id: str,
    sample_path: Path,
    review_sha256: str,
) -> MatchedSetConfigV1:
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        sample = connection.execute(
            "SELECT s.*,r.dataset_name,r.version AS release_version,r.file_sha256,"
            "lp.label_policy_id,lp.name AS policy_name,lp.version AS policy_version,"
            "lp.rule_sha256,lp.reviewed,lp.review_version,lp.review_sha256,"
            "pt.name AS template_name,pt.version AS template_version,pt.body_sha256 "
            "FROM candidate_sample s "
            "JOIN dataset_release r ON r.release_id=s.release_id "
            "JOIN label_policy lp ON lp.label_policy_id=s.label_policy_id "
            "JOIN presentation_template pt ON pt.template_id=s.template_id "
            "WHERE s.sample_id=?",
            (sample_id,),
        ).fetchone()
        templates = {
            str(row["name"]): row
            for row in connection.execute(
                "SELECT template_id,name,version,body_sha256 FROM instruction_template "
                "WHERE version=?",
                (INSTRUCTION_TEMPLATE_VERSION,),
            )
        }
    finally:
        connection.close()
    if sample is None:
        raise RuntimeError(f"default pipeline sample {sample_id} disappeared")
    required_templates = {
        "setup",
        "statement",
        "ballot",
        "correction",
        "result",
        "final-result",
    }
    if set(templates) != required_templates:
        raise RuntimeError(
            "Default pipeline requires exactly the six current instruction templates; "
            f"found {sorted(templates)} for version {INSTRUCTION_TEMPLATE_VERSION}."
        )
    route = resolve_route(ModelId.GEMMA_4_E2B_IT, ProviderId.LOCAL, QuantizationId.BF16)
    if not isinstance(route, LocalTransformersRoute):
        raise AssertionError("the reviewed default Gemma route must be local")

    def selector(name: str) -> dict[str, str]:
        row = templates[name]
        return {
            "template_id": str(row["template_id"]),
            "name": str(row["name"]),
            "version": str(row["version"]),
            "expected_sha256": str(row["body_sha256"]),
        }

    return MatchedSetConfigV1.model_validate(
        {
            "schema_version": "qv-run-config/v1",
            "canonical_json_version": "qv-canonical-json/v1",
            "prompt_encoding_version": "qv-prompt/v1",
            "seed_version": "qv-seed/v1",
            "sample": {
                "sample_id": sample_id,
                "artifact_path": sample_path.resolve(),
                "expected_sha256": sample["artifact_sha256"],
                "release": {
                    "release_id": sample["release_id"],
                    "dataset_name": sample["dataset_name"],
                    "version": sample["release_version"],
                    "expected_sha256": sample["file_sha256"],
                },
                "label_policy": {
                    "label_policy_id": sample["label_policy_id"],
                    "name": sample["policy_name"],
                    "version": sample["policy_version"],
                    "expected_sha256": sample["rule_sha256"],
                    "reviewed": bool(sample["reviewed"]),
                    "review_version": sample["review_version"],
                    "review_sha256": sample["review_sha256"],
                },
                "presentation_template": {
                    "template_id": sample["template_id"],
                    "name": sample["template_name"],
                    "version": sample["template_version"],
                    "expected_sha256": sample["body_sha256"],
                },
            },
            "route": {
                "model_id": route.model_id.value,
                "provider_id": route.provider_id.value,
                "quantization_id": route.quantization_id.value,
                "runtime_id": route.runtime_id.value,
                "artifact_repository": route.artifact.repository,
                "artifact_revision": route.artifact.revision,
                "tokenizer_repository": route.artifact.repository,
                "tokenizer_revision": route.artifact.revision,
                "dtype": "bf16",
            },
            "prompts": {
                "setup": selector("setup"),
                "statement": selector("statement"),
                "ballot": selector("ballot"),
                "correction": selector("correction"),
                "result": selector("result"),
                "final_result": selector("final-result"),
                "reviewed": True,
                "review_version": _REVIEW_VERSION,
                "review_sha256": review_sha256,
            },
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 10,
                "max_new_tokens": 2048,
            },
            "ballot_retry": {"max_corrections": 3},
            "statement_retry": {"max_corrections": 3},
            "runtime_retry": {
                "max_failures_per_execution": 3,
                "initial_backoff_ms": 1000,
                "multiplier": 2.0,
                "max_backoff_ms": 2000,
            },
            "master_seed": args.master_seed,
            "voter_count": args.voters,
            "credit_budget": 100,
            "sampler_policy": "balanced-matched/v1",
            "presentation_policy": "setup-once-ids-later/v1",
            "tie_policy": "uniform-seeded/v1",
            "action_format": "json-with-rationale/v1",
            "execution_class": "pilot",
        }
    )


def _checkpoint(path: Path, manifest: dict[str, Any], **values: object) -> None:
    manifest.update(values)
    _write_json(path, manifest)


def _initial_manifest(
    args: argparse.Namespace, review_sha256: str, database_identity: str
) -> dict[str, Any]:
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.is_file():
        raise ValueError(f"default pipeline dataset does not exist: {dataset_path}")
    return {
        "schema_version": _PIPELINE_VERSION,
        "database": str(args.db.resolve()),
        "database_identity": database_identity,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _file_sha256(dataset_path),
        "dataset_version": args.dataset_version,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "master_seed": args.master_seed,
        "voters": args.voters,
        "cache_dir": str(args.cache_dir.resolve()),
        "device": args.device.value,
        "review_artifact": str(args.review_artifact.resolve()),
        "review_sha256": review_sha256,
        "stage": "initialized",
    }


def _load_manifest(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Default pipeline cannot resume because manifest {path} is unreadable or "
            "invalid JSON. Restore it from the completed setup step or start with a new "
            "output directory and database."
        ) from error
    if not isinstance(value, dict) or value.get("schema_version") != _PIPELINE_VERSION:
        raise ValueError(
            f"Default pipeline manifest {path} is not {_PIPELINE_VERSION}. Use the code "
            "version that created it or choose a new output directory and database."
        )
    expected = {
        "database": str(args.db.resolve()),
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_version": args.dataset_version,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "master_seed": args.master_seed,
        "voters": args.voters,
        "cache_dir": str(args.cache_dir.resolve()),
        "device": args.device.value,
    }
    drift = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if drift:
        raise ValueError(
            f"Default pipeline refused resume because invocation settings drifted: {drift}. "
            "Rerun with the original settings or choose a new output directory and database."
        )
    database_identity = _database_identity(args.db, create=False)
    if database_identity != value.get("database_identity"):
        raise ValueError(
            f"Default pipeline refused resume because database identity changed for {args.db}. "
            "Restore the original database rather than reusing this output directory."
        )
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.is_file():
        raise ValueError(
            f"Default pipeline cannot resume because dataset {dataset_path} is missing. "
            "Restore the original dataset bytes or start a new pipeline."
        )
    dataset_sha256 = _file_sha256(dataset_path)
    if value.get("dataset_sha256") != dataset_sha256:
        raise ValueError(
            f"Default pipeline refused resume because dataset {dataset_path} changed. "
            "Restore the original dataset bytes or start a new pipeline."
        )
    review_path = args.review_artifact.resolve()
    if not review_path.is_file():
        raise ValueError(
            f"Default pipeline review artifact {review_path} is missing. Restore the "
            "repository-tracked approval record before resuming."
        )
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    if value.get("review_sha256") != review_sha256:
        raise ValueError(
            f"Default pipeline refused resume because review artifact {review_path} changed. "
            "Use the original reviewed bytes or start a new pipeline."
        )
    print(f"pipeline_manifest={path} status=resuming")
    return value


def _ensure_release(
    args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path
) -> str:
    source = args.dataset_path.resolve()
    if not source.is_file():
        raise ValueError(f"default pipeline dataset does not exist: {source}")
    source_sha256 = _file_sha256(source)
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        if manifest.get("release_id") is not None:
            row = connection.execute(
                "SELECT release_id,file_sha256,version FROM dataset_release "
                "WHERE release_id=?",
                (manifest["release_id"],),
            ).fetchone()
            if row is not None and row["version"] != args.dataset_version:
                raise ValueError(
                    "Default pipeline found a replacement database release for the "
                    "manifest-bound release ID. Restore the original database."
                )
        else:
            row = connection.execute(
                "SELECT release_id,file_sha256,version FROM dataset_release "
                "WHERE dataset_name='ConvAbuse' AND version=?",
                (args.dataset_version,),
            ).fetchone()
    finally:
        connection.close()
    if row is None:
        if manifest.get("release_id") is not None:
            raise ValueError(
                "Default pipeline manifest references a missing release ID. Refusing to "
                "adopt a replacement database object; restore the original database."
            )
        ingest = _invoke(
            args,
            [
                "catalog",
                "ingest",
                "--dataset-path",
                str(source),
                "--dataset-version",
                args.dataset_version,
                "--rule",
                _LABEL_POLICY_VERSION,
            ],
        )
        release_id = _value(ingest, "release_id")
    else:
        if row["file_sha256"] != source_sha256:
            raise ValueError(
                f"Default pipeline found ConvAbuse/{args.dataset_version} with source hash "
                f"{row['file_sha256']}, not current {source_sha256}. Use the original source "
                "bytes or choose a new --dataset-version and output directory."
            )
        release_id = str(row["release_id"])
        if (
            manifest.get("release_id") is not None
            and release_id != manifest["release_id"]
        ):
            raise ValueError(
                "Default pipeline refused to adopt a database release with a different "
                "manifest-bound release ID. Restore the original database."
            )
        print(f"release_id={release_id} status=existing")
    _checkpoint(
        manifest_path,
        manifest,
        stage="release",
        release_id=release_id,
        dataset_sha256=source_sha256,
    )
    return release_id


def _ensure_templates(args: argparse.Namespace, release_id: str) -> str:
    card_hash = hashlib.sha256(DEFAULT_PRESENTATION_TEMPLATE_BODY.encode()).hexdigest()
    with acquire_writer_lock(
        args.db, command="pipeline-default-templates"
    ) as writer_lock:
        with open_sqlite_store(
            args.db, writer_lock=writer_lock, require_writer_lock=True
        ) as store:
            card = store.connection.execute(
                "SELECT template_id,body_sha256 FROM presentation_template "
                "WHERE name=? AND version=?",
                (
                    DEFAULT_PRESENTATION_TEMPLATE_NAME,
                    DEFAULT_PRESENTATION_TEMPLATE_VERSION,
                ),
            ).fetchone()
            if card is None:
                card_template_id = str(
                    store.register_template(
                        DEFAULT_PRESENTATION_TEMPLATE_NAME,
                        DEFAULT_PRESENTATION_TEMPLATE_VERSION,
                        DEFAULT_PRESENTATION_TEMPLATE_BODY,
                    )
                )
            elif card["body_sha256"] != card_hash:
                raise ValueError(
                    "candidate-card/v3 hash drifted from the pipeline default"
                )
            else:
                card_template_id = str(card["template_id"])
            for kind, body in TEMPLATE_BODIES.items():
                digest = hashlib.sha256(body.encode()).hexdigest()
                row = store.connection.execute(
                    "SELECT template_id,body_sha256 FROM instruction_template "
                    "WHERE name=? AND version=?",
                    (kind.value, INSTRUCTION_TEMPLATE_VERSION),
                ).fetchone()
                if row is None:
                    store.register_template(kind, INSTRUCTION_TEMPLATE_VERSION, body)
                elif row["body_sha256"] != digest:
                    raise ValueError(
                        f"{kind.value}/{INSTRUCTION_TEMPLATE_VERSION} hash drifted from "
                        "the pipeline default"
                    )
            candidate_count = store.connection.execute(
                "SELECT COUNT(*) FROM candidate WHERE release_id=?", (release_id,)
            ).fetchone()[0]
            presentation_count = store.connection.execute(
                "SELECT COUNT(*) FROM candidate_presentation cp JOIN candidate c "
                "ON c.candidate_id=cp.candidate_id "
                "WHERE c.release_id=? AND cp.template_id=?",
                (release_id, card_template_id),
            ).fetchone()[0]
            if presentation_count == 0:
                render_release_candidate_cards(
                    store, ReleaseId(release_id), TemplateId(card_template_id)
                )
            elif presentation_count != candidate_count:
                raise RuntimeError(
                    f"release {release_id} has {presentation_count}/{candidate_count} "
                    "candidate-card presentations; restore the catalog before resuming"
                )
    print(f"candidate-card={card_template_id} status=ready")
    return card_template_id


def _ensure_sample(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    release_id: str,
    card_template_id: str,
) -> tuple[str, Path]:
    sample_path = args.output_dir.resolve() / "sample.json"
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        if manifest.get("sample_id") is not None:
            rows = connection.execute(
                "SELECT sample_id,status,release_id,template_id FROM candidate_sample "
                "WHERE sample_id=?",
                (manifest["sample_id"],),
            ).fetchall()
            if rows and (
                rows[0]["release_id"] != release_id
                or rows[0]["template_id"] != card_template_id
            ):
                raise ValueError(
                    "Default pipeline refused a replacement database sample with a "
                    "different release or template identity. Restore the original database."
                )
        else:
            rows = connection.execute(
                "SELECT sample_id,status,release_id,template_id FROM candidate_sample "
                "WHERE release_id=? AND template_id=? AND sampler_policy='balanced-matched' "
                "AND sampler_seed=? AND size=?",
                (
                    release_id,
                    card_template_id,
                    seed_to_blob(args.sample_seed),
                    args.sample_size,
                ),
            ).fetchall()
    finally:
        connection.close()
    if len(rows) > 1:
        raise RuntimeError(
            "Default pipeline found multiple samples for the same immutable release, "
            "template, seed, and size. Select one through the lower-level workflow."
        )
    if rows:
        sample_id = str(rows[0]["sample_id"])
        print(f"sample_id={sample_id} status={rows[0]['status']}")
    else:
        if manifest.get("sample_id") is not None:
            raise ValueError(
                "Default pipeline manifest references a missing sample ID. Refusing to "
                "adopt a replacement database object; restore the original database."
            )
        created = _invoke(
            args,
            [
                "sample",
                "create",
                "--release-id",
                release_id,
                "--template-id",
                card_template_id,
                "--size",
                str(args.sample_size),
                "--seed",
                str(args.sample_seed),
            ],
        )
        sample_id = _value(created, "sample_id")
    _invoke(
        args,
        [
            "sample",
            "freeze",
            "--sample-id",
            sample_id,
            "--out",
            str(sample_path),
        ],
    )
    _invoke(
        args,
        [
            "sample",
            "verify",
            "--sample-id",
            sample_id,
            "--artifact",
            str(sample_path),
        ],
    )
    _checkpoint(
        manifest_path,
        manifest,
        stage="sample",
        sample_id=sample_id,
        sample_path=str(sample_path),
        sample_sha256=_file_sha256(sample_path),
    )
    return sample_id, sample_path


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=lambda item: (
            {"__bytes__": bytes(item).hex()} if isinstance(item, bytes) else str(item)
        ),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runs_for_matched_set(database: Path, matched_set_id: str) -> list[dict[str, str]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT run_id,arm,regime FROM experiment_run WHERE matched_set_id=? "
            "ORDER BY arm,regime",
            (matched_set_id,),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _ensure_matched_set(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    config: MatchedSetConfigV1,
    config_path: Path,
) -> tuple[str, list[dict[str, str]]]:
    config_value = config.model_dump(mode="json")
    config_hash = _canonical_hash(config_value)
    _write_json(config_path, config_value)
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT m.matched_set_id FROM experiment_config_record c "
            "JOIN matched_set m ON m.config_id=c.config_id WHERE c.config_hash=?",
            (config_hash,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        matched = _invoke(args, ["matched-set", "create", "--config", str(config_path)])
        matched_set_id = _value(matched, "matched_set_id")
    else:
        matched_set_id = str(row["matched_set_id"])
        print(f"matched_set_id={matched_set_id} status=existing")
    runs = _runs_for_matched_set(args.db, matched_set_id)
    expected_pairs = {
        (arm.value, regime.value) for arm in ElicitationArm for regime in VotingRegime
    }
    actual_pairs = {(run["arm"], run["regime"]) for run in runs}
    if len(runs) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise RuntimeError(
            f"Default pipeline matched set {matched_set_id} has invalid run matrix "
            f"{sorted(actual_pairs)}. Restore its six atomic runs before resuming."
        )
    _checkpoint(
        manifest_path,
        manifest,
        stage="matched-set",
        config_path=str(config_path),
        config_sha256=_file_sha256(config_path),
        config_hash=config_hash,
        matched_set_id=matched_set_id,
        runs=runs,
    )
    return matched_set_id, runs


def _validate_ready_manifest(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> tuple[str, list[dict[str, str]]]:
    required = {
        "release_id",
        "dataset_sha256",
        "sample_id",
        "sample_path",
        "sample_sha256",
        "config_path",
        "config_sha256",
        "config_hash",
        "matched_set_id",
        "runs",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"pipeline manifest is incomplete after setup: {missing}")
    sample_path = Path(str(manifest["sample_path"]))
    config_path = Path(str(manifest["config_path"]))
    file_checks = {
        "sample_sha256": (
            sample_path,
            str(manifest["sample_sha256"]),
        ),
        "config_sha256": (
            config_path,
            str(manifest["config_sha256"]),
        ),
    }
    for name, (path, expected_hash) in file_checks.items():
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(
                f"Default pipeline resume refused changed or missing {name} artifact {path}. "
                "Restore the manifest-bound bytes or start a new pipeline."
            )
    config = MatchedSetConfigV1.from_json_file(config_path)
    if _canonical_hash(config.model_dump(mode="json")) != manifest["config_hash"]:
        raise ValueError("pipeline config canonical hash differs from its manifest")
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT m.matched_set_id,c.config_hash,ed.sample_id,ed.release_id,r.file_sha256,"
            "lp.review_version,lp.review_sha256,s.artifact_sha256 "
            "FROM matched_set m JOIN experiment_config_record c ON c.config_id=m.config_id "
            "JOIN experiment_definition ed ON ed.definition_id=c.definition_id "
            "JOIN candidate_sample s ON s.sample_id=ed.sample_id "
            "JOIN dataset_release r ON r.release_id=ed.release_id "
            "JOIN label_policy lp ON lp.label_policy_id=ed.label_policy_id "
            "WHERE m.matched_set_id=?",
            (manifest["matched_set_id"],),
        ).fetchone()
    finally:
        connection.close()
    expected = (
        manifest["matched_set_id"],
        manifest["config_hash"],
        manifest["sample_id"],
        manifest["release_id"],
        manifest["dataset_sha256"],
        _REVIEW_VERSION,
        manifest["review_sha256"],
        manifest["sample_sha256"],
    )
    actual = tuple(row) if row is not None else None
    if actual != expected:
        raise ValueError(
            "Default pipeline resume refused database/manifest identity mismatch: "
            f"persisted={actual!r}, expected={expected!r}. Restore the original database "
            "or start with a new output directory."
        )
    current_dataset_sha256 = _file_sha256(args.dataset_path.resolve())
    if current_dataset_sha256 != manifest["dataset_sha256"]:
        raise ValueError(
            "Default pipeline resume refused dataset bytes that differ from the "
            "manifest-bound release. Restore the original dataset."
        )
    matched_set_id = str(manifest["matched_set_id"])
    runs = _runs_for_matched_set(args.db, matched_set_id)
    if runs != manifest["runs"]:
        raise ValueError(
            f"Default pipeline resume refused changed run IDs for {matched_set_id}."
        )
    return matched_set_id, runs


def _directory_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        return {}
    return {
        str(item.relative_to(path)): _file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _model_provenance(
    model_path: Path, *, repository: str, revision: str
) -> dict[str, Any]:
    """Describe the complete, immutable snapshot used by the pipeline."""
    if not model_path.is_dir():
        raise ValueError(
            f"Default pipeline model snapshot is not a directory: {model_path}"
        )
    files = _directory_hashes(model_path)
    missing = sorted(_MODEL_SNAPSHOT_FILES - files.keys())
    unexpected = sorted(files.keys() - _MODEL_SNAPSHOT_FILES)
    unexpected_weights = sorted(
        name
        for name in files
        if name.endswith(".safetensors") and name != "model.safetensors"
    )
    if unexpected_weights or "model.safetensors.index.json" in files:
        raise ValueError(
            f"Default pipeline model snapshot {model_path} uses an unsupported sharded "
            f"or alternate safetensors layout: {unexpected_weights}. The pinned Gemma "
            "route requires exactly model.safetensors."
        )
    if missing:
        raise ValueError(
            f"Default pipeline model snapshot {model_path} is incomplete; missing "
            f"essential files {missing}. Refresh the pinned model cache and retry."
        )
    if unexpected:
        raise ValueError(
            f"Default pipeline model snapshot {model_path} has unexpected files "
            f"{unexpected}. The reviewed Gemma route requires the exact pinned "
            "single-file snapshot layout. Use the unmodified pinned snapshot."
        )
    return {
        "repository": repository,
        "revision": revision,
        "resolved_path": str(model_path.resolve()),
        "files": files,
    }


def _bind_model_provenance(
    args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path
) -> None:
    route = resolve_route(ModelId.GEMMA_4_E2B_IT, ProviderId.LOCAL, QuantizationId.BF16)
    if not isinstance(route, LocalTransformersRoute):
        raise AssertionError("the reviewed default Gemma route must be local")
    model_path = download_transformers_artifact(route, args.cache_dir)
    provenance = _model_provenance(
        model_path,
        repository=route.artifact.repository,
        revision=route.artifact.revision,
    )
    recorded = manifest.get("model_provenance")
    if recorded is not None and recorded != provenance:
        raise ValueError(
            "Default pipeline refused model artifact provenance drift on resume. "
            "The pinned repository, revision, resolved snapshot path, or snapshot file "
            "hashes changed; restore the original cache and retry."
        )
    if recorded is None:
        _checkpoint(manifest_path, manifest, model_provenance=provenance)
    print(f"model_artifact={model_path} status=verified")


def _matched_set_source_fingerprint(database: Path, matched_set_id: str) -> str:
    """Hash stable rows belonging to this matched set and its immutable inputs."""
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        matched = connection.execute(
            "SELECT * FROM matched_set WHERE matched_set_id=?", (matched_set_id,)
        ).fetchone()
        if matched is None:
            raise ValueError(f"Cannot fingerprint missing matched set {matched_set_id}")
        runs = connection.execute(
            "SELECT * FROM experiment_run WHERE matched_set_id=? ORDER BY run_id",
            (matched_set_id,),
        ).fetchall()
        identifiers: dict[str, set[str]] = {
            "matched_set_id": {matched_set_id},
            "run_id": {str(row["run_id"]) for row in runs},
        }
        for column in (
            "config_id",
            "definition_id",
            "sample_id",
            "release_id",
            "label_policy_id",
            "template_id",
            "model_id",
        ):
            identifiers[column] = set()
        identifiers["config_id"].add(str(matched["config_id"]))
        config = connection.execute(
            "SELECT * FROM experiment_config_record WHERE config_id=?",
            (matched["config_id"],),
        ).fetchone()
        if config is not None:
            identifiers["definition_id"].add(str(config["definition_id"]))
        definition = connection.execute(
            "SELECT * FROM experiment_definition WHERE definition_id=?",
            (next(iter(identifiers["definition_id"]), ""),),
        ).fetchone()
        if definition is not None:
            for column in (
                "sample_id",
                "release_id",
                "label_policy_id",
                "template_id",
                "model_id",
            ):
                if column in definition.keys():
                    identifiers[column].add(str(definition[column]))
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        payload: list[object] = []
        for table_row in tables:
            table = str(table_row[0])
            columns = [
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            ]
            predicates: list[str] = []
            values: list[str] = []
            for column in columns:
                values_for_column = identifiers.get(column, set())
                if values_for_column:
                    placeholders = ",".join("?" for _ in values_for_column)
                    predicates.append(f'"{column}" IN ({placeholders})')
                    values.extend(sorted(values_for_column))
            if not predicates:
                continue
            rows = connection.execute(
                f'SELECT * FROM "{table}" WHERE {" OR ".join(predicates)} ORDER BY rowid',
                values,
            ).fetchall()
            payload.append(
                {
                    "table": table,
                    "columns": columns,
                    "rows": [dict(row) for row in rows],
                }
            )
        return _canonical_hash(payload)
    finally:
        connection.close()


def _ensure_derived_artifact(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    output: Path,
    manifest_key: str,
    binding_key: str,
    source_binding: dict[str, Any],
    command: list[str],
) -> bool:
    recorded = manifest.get(manifest_key)
    recorded_binding = manifest.get(binding_key)
    current = _directory_hashes(output)
    if (
        isinstance(recorded, dict)
        and recorded
        and recorded_binding == source_binding
        and current == recorded
    ):
        print(f"{manifest_key}={output} status=verified")
        return False
    if output.exists():
        shutil.rmtree(output)
    _invoke(args, command)
    generated = _directory_hashes(output)
    if not generated:
        raise RuntimeError(f"pipeline command produced no files in {output}")
    _checkpoint(
        manifest_path,
        manifest,
        **{manifest_key: generated, binding_key: source_binding},
    )
    return True


def _run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir.resolve() / "manifest.json"
    review_path = args.review_artifact.resolve()
    if not review_path.is_file():
        raise ValueError(
            f"Default pipeline review artifact {review_path} is missing. Restore the "
            "repository-tracked approval record before creating a pilot."
        )
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    manifest_exists = manifest_path.is_file()
    if manifest_exists and not args.db.is_file():
        raise ValueError(
            f"Default pipeline manifest {manifest_path} references missing database "
            f"{args.db}. Restore that database or start with a new output directory."
        )
    if manifest_exists:
        manifest = _load_manifest(args, manifest_path)
    else:
        _invoke(args, ["migrate"])
        database_identity = _database_identity(args.db, create=True)
        if database_identity is None:
            raise RuntimeError(
                f"Pipeline could not establish durable identity for {args.db}"
            )
        manifest = _initial_manifest(args, review_sha256, database_identity)
        _write_json(manifest_path, manifest)
        print(f"pipeline_manifest={manifest_path} status=initialized")

    if "matched_set_id" in manifest:
        matched_set_id, runs = _validate_ready_manifest(args, manifest)
    else:
        release_id = _ensure_release(args, manifest, manifest_path)
        card_template_id = _ensure_templates(args, release_id)
        _checkpoint(
            manifest_path,
            manifest,
            stage="templates",
            card_template_id=card_template_id,
        )
        sample_id, sample_path = _ensure_sample(
            args,
            manifest,
            manifest_path,
            release_id=release_id,
            card_template_id=card_template_id,
        )
        _approve_label_policy(args.db, review_sha256, command="pipeline-default-review")
        _checkpoint(manifest_path, manifest, stage="reviewed")
        config_path = args.output_dir.resolve() / "run-config.json"
        config = _build_config(
            args,
            sample_id=sample_id,
            sample_path=sample_path,
            review_sha256=review_sha256,
        )
        matched_set_id, runs = _ensure_matched_set(
            args,
            manifest,
            manifest_path,
            config=config,
            config_path=config_path,
        )
        _validate_ready_manifest(args, manifest)

    if args.generator_factory is None:
        args.generator_factory = _shared_default_generator_factory(args)
        _bind_model_provenance(args, manifest, manifest_path)
    for run in runs:
        _invoke(
            args,
            [
                "run",
                "--run-id",
                run["run_id"],
                "--cache-dir",
                str(args.cache_dir),
                "--device",
                args.device.value,
            ],
        )
        _invoke(args, ["verify", "--run-id", run["run_id"]])

    source_binding = {
        "matched_set_id": matched_set_id,
        "run_ids": [run["run_id"] for run in runs],
        "source_fingerprint": _matched_set_source_fingerprint(args.db, matched_set_id),
    }
    export_dir = args.output_dir.resolve() / "export"
    export_changed = _ensure_derived_artifact(
        args,
        manifest,
        manifest_path,
        output=export_dir,
        manifest_key="export_files",
        binding_key="export_binding",
        source_binding=source_binding,
        command=[
            "export",
            "--matched-set",
            matched_set_id,
            "--out",
            str(export_dir),
        ],
    )
    if export_changed:
        manifest.pop("plot_files", None)
        _write_json(manifest_path, manifest)
    plot_dir = args.output_dir.resolve() / "plots"
    plot_binding = {
        "export_binding": source_binding,
        "export_files": _directory_hashes(export_dir),
    }
    _ensure_derived_artifact(
        args,
        manifest,
        manifest_path,
        output=plot_dir,
        manifest_key="plot_files",
        binding_key="plot_binding",
        source_binding=plot_binding,
        command=["plot", "--export-dir", str(export_dir), "--out", str(plot_dir)],
    )
    _checkpoint(manifest_path, manifest, stage="complete")
    print(
        f"pipeline=complete matched_set_id={matched_set_id} "
        f"manifest={manifest_path} export={export_dir} plots={plot_dir}"
    )
    return 0


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    pipeline = subparsers.add_parser(
        "pipeline", help="run the reviewed default six-condition pilot"
    )
    pipeline_sub = pipeline.add_subparsers(required=True)
    run = pipeline_sub.add_parser(
        "run", help="create, execute, verify, export, and plot the default pilot"
    )
    run.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("emotion_probing/data/ConvAbuseEMNLPfull.csv"),
    )
    run.add_argument("--dataset-version", default="convabuse-emnlp-full/default-v3")
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quadratic_voting/data/default-pilot"),
    )
    run.add_argument(
        "--review-artifact",
        type=Path,
        default=Path("quadratic_voting/DEFAULT_PILOT_REVIEW_V2.md"),
    )
    run.add_argument("--sample-size", type=_positive, default=10)
    run.add_argument("--sample-seed", type=_seed, default=_DEFAULT_SEED)
    run.add_argument("--master-seed", type=_seed, default=_DEFAULT_SEED)
    run.add_argument("--voters", type=_positive, default=3)
    run.add_argument("--cache-dir", type=Path, default=Path(HF_HUB_CACHE))
    run.add_argument("--device", type=Device, choices=list(Device), default=Device.CUDA)
    run.set_defaults(handler=_run, mutates_db=False)
