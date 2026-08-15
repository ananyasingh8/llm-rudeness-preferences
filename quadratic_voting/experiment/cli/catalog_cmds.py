"""Catalog ingestion and immutable template registration CLI commands."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from quadratic_voting.experiment.catalog import (
    DEFAULT_PRESENTATION_TEMPLATE_BODY,
    DEFAULT_PRESENTATION_TEMPLATE_NAME,
    DEFAULT_PRESENTATION_TEMPLATE_VERSION,
    RudenessDerivationRule,
    ingest_convabuse,
)
from quadratic_voting.experiment.store import open_sqlite_store
from quadratic_voting.experiment.transcript import TEMPLATE_BODIES
from quadratic_voting.experiment.types import TemplateKind


def _rule(value: str) -> RudenessDerivationRule:
    try:
        return RudenessDerivationRule(value)
    except ValueError as error:
        choices = ", ".join(rule.value for rule in RudenessDerivationRule)
        raise argparse.ArgumentTypeError(
            f"unknown rudeness derivation rule {value!r}; choose one versioned rule: {choices}"
        ) from error


def _ingest(args: argparse.Namespace) -> int:
    with open_sqlite_store(
        args.db, writer_lock=args.writer_lock, require_writer_lock=True
    ) as store:
        release_id = ingest_convabuse(
            store, args.dataset_path, args.dataset_version, args.rule
        )
    print(f"release_id={release_id}")
    return 0


def _templates(args: argparse.Namespace) -> int:
    with open_sqlite_store(
        args.db, writer_lock=args.writer_lock, require_writer_lock=True
    ) as store:
        try:
            presentation_id = store.register_template(
                DEFAULT_PRESENTATION_TEMPLATE_NAME,
                DEFAULT_PRESENTATION_TEMPLATE_VERSION,
                DEFAULT_PRESENTATION_TEMPLATE_BODY,
            )
        except sqlite3.IntegrityError:
            row = store.connection.execute(
                "SELECT template_id FROM presentation_template WHERE name=? AND version=?",
                (
                    DEFAULT_PRESENTATION_TEMPLATE_NAME,
                    DEFAULT_PRESENTATION_TEMPLATE_VERSION,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "Template registration failed because candidate-card/v1 conflicted but "
                    "could not be resolved in presentation_template. Inspect and restore the "
                    "SQLite catalog, then retry catalog-templates --register."
                )
            presentation_id = row[0]
        registered = [
            (kind, store.register_template(kind, "v1", TEMPLATE_BODIES[kind]))
            for kind in TemplateKind
        ]
    instruction_output = " ".join(
        f"{kind.value}={template_id}" for kind, template_id in registered
    )
    print(f"candidate-card={presentation_id} {instruction_output}")
    return 0


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    catalog = subparsers.add_parser("catalog", help="immutable dataset catalog")
    catalog_sub = catalog.add_subparsers(required=True)
    ingest = catalog_sub.add_parser(
        "ingest", help="ingest one immutable ConvAbuse release"
    )
    ingest.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("emotion_probing/data/ConvAbuseEMNLPfull.csv"),
    )
    ingest.add_argument("--dataset-version", required=True)
    ingest.add_argument(
        "--rule",
        type=_rule,
        default=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE,
        choices=list(RudenessDerivationRule),
    )
    ingest.set_defaults(handler=_ingest, mutates_db=True)

    template = subparsers.add_parser("template", help="reviewed prompt templates")
    template_sub = template.add_subparsers(required=True)
    templates = template_sub.add_parser(
        "register", help="register the six immutable instruction templates"
    )
    templates.set_defaults(handler=_templates, mutates_db=True)

    legacy_ingest = subparsers.add_parser("catalog-ingest", help=argparse.SUPPRESS)
    for action in ingest._actions[1:]:
        if action.dest == "dataset_path":
            legacy_ingest.add_argument(
                "--dataset-path", type=Path, default=action.default
            )
        elif action.dest == "dataset_version":
            legacy_ingest.add_argument("--dataset-version", required=True)
        elif action.dest == "rule":
            legacy_ingest.add_argument(
                "--rule",
                type=_rule,
                default=action.default,
                choices=list(RudenessDerivationRule),
            )
    legacy_ingest.set_defaults(handler=_ingest, mutates_db=True)
    legacy_templates = subparsers.add_parser(
        "catalog-templates", help=argparse.SUPPRESS
    )
    legacy_templates.add_argument("--register", action="store_true", required=True)
    legacy_templates.set_defaults(handler=_templates, mutates_db=True)
