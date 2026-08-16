"""Balanced sample creation, freezing, and validation CLI commands."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from quadratic_voting.experiment.sampling import (
    candidates_by_label,
    candidates_by_severity_level,
    create_balanced_sample,
    create_level_stratified_sample,
)
from quadratic_voting.experiment.store import open_sqlite_store
from quadratic_voting.experiment.types import (
    ReleaseId,
    SampleId,
    SamplerPolicy,
    TemplateId,
)


def _sample_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer; expected a sample size of at least two"
        ) from error
    if parsed < 2:
        raise argparse.ArgumentTypeError(
            f"{value!r} is below two; use an integer sample size of at least two"
        )
    return parsed


def _non_negative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer; expected a non-negative RNG seed"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} is negative; use zero or a positive integer RNG seed"
        )
    return parsed


def _create(args: argparse.Namespace) -> int:
    release_id = ReleaseId(args.release_id)
    policy = SamplerPolicy(args.policy)
    with open_sqlite_store(
        args.db, writer_lock=args.writer_lock, require_writer_lock=True
    ) as store:
        if policy is SamplerPolicy.LEVEL_STRATIFIED:
            # The level-stratified policy draws exactly one candidate per ConvAbuse
            # severity level (five candidates); --size is not consulted.
            sample_id = create_level_stratified_sample(
                store,
                release_id,
                TemplateId(args.template_id),
                seed=args.seed,
                candidates_by_level=candidates_by_severity_level(store, release_id),
            )
        else:
            sample_id = create_balanced_sample(
                store,
                release_id,
                TemplateId(args.template_id),
                size=args.size,
                seed=args.seed,
                candidates_by_label=candidates_by_label(store, release_id),
            )
    print(f"sample_id={sample_id} status=DRAFT")
    return 0


def _freeze(args: argparse.Namespace) -> int:
    with open_sqlite_store(
        args.db, writer_lock=args.writer_lock, require_writer_lock=True
    ) as store:
        store.freeze_sample(SampleId(args.sample_id), args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"artifact={args.out} sha256={digest}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    # Imported lazily to avoid a registration cycle while sharing the no-migration
    # read-only store construction used by inspect and verify.
    from quadratic_voting.experiment.cli.run_cmds import _open_readonly_store

    with _open_readonly_store(args.db) as store:
        sample_id = store.validate_sample(args.file)
    if str(sample_id) != args.sample_id:
        raise ValueError(
            f"Sample verification refused artifact {args.file} because it belongs to "
            f"sample {sample_id}, not requested sample {args.sample_id}. Verification "
            "failed in cli.sample_cmds before reporting success. Supply the matching "
            "sample ID and artifact together."
        )
    print(f"sample_id={sample_id}")
    return 0


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    sample = subparsers.add_parser("sample", help="candidate sample lifecycle")
    sample_sub = sample.add_subparsers(required=True)
    create = sample_sub.add_parser(
        "create",
        help="create a deterministic balanced or level-stratified DRAFT sample",
    )
    create.add_argument("--release-id", required=True)
    create.add_argument("--template-id", required=True)
    create.add_argument("--size", type=_sample_size, default=50)
    create.add_argument("--seed", type=_non_negative_integer, required=True)
    create.add_argument(
        "--policy",
        choices=[
            SamplerPolicy.BALANCED_MATCHED.value,
            SamplerPolicy.LEVEL_STRATIFIED.value,
        ],
        default=SamplerPolicy.BALANCED_MATCHED.value,
        help="balanced two-stratum sample, or one candidate per severity level",
    )
    create.set_defaults(handler=_create, mutates_db=True)

    freeze = sample_sub.add_parser(
        "freeze", help="freeze a DRAFT sample as a bare candidate-ID array"
    )
    freeze.add_argument("--sample-id", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    freeze.set_defaults(handler=_freeze, mutates_db=True)

    verify = sample_sub.add_parser(
        "verify", help="verify frozen sample hash and ordered membership"
    )
    verify.add_argument("--sample-id", required=True)
    verify.add_argument("--artifact", dest="file", type=Path, required=True)
    verify.set_defaults(handler=_validate)
