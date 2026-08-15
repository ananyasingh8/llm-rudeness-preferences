from __future__ import annotations

import copy
import json
import unittest

from pydantic import ValidationError

from quadratic_voting.experiment.config import MatchedSetConfigV1


def valid_config() -> dict[str, object]:
    digest = "a" * 64

    def template(kind: str) -> dict[str, str]:
        return {
            "template_id": f"template-{kind}",
            "name": kind,
            "version": "v1",
            "expected_sha256": digest,
        }

    return {
        "schema_version": "qv-run-config/v1",
        "canonical_json_version": "qv-canonical-json/v1",
        "prompt_encoding_version": "qv-prompt/v1",
        "seed_version": "qv-seed/v1",
        "sample": {
            "sample_id": "sample-1",
            "artifact_path": "samples/one.json",
            "expected_sha256": digest,
            "release": {
                "release_id": "release-1",
                "dataset_name": "ConvAbuse",
                "version": "v1",
                "expected_sha256": digest,
            },
            "label_policy": {
                "label_policy_id": "label-1",
                "name": "majority",
                "version": "v1",
                "expected_sha256": digest,
                "reviewed": True,
                "review_version": "review/v1",
                "review_sha256": digest,
            },
            "presentation_template": template("candidate-card"),
        },
        "route": {
            "model_id": "gemma",
            "provider_id": "transformers",
            "quantization_id": "bf16",
            "runtime_id": "local",
            "artifact_repository": "repo",
            "artifact_revision": "revision",
            "tokenizer_repository": "repo",
            "tokenizer_revision": "revision",
            "dtype": "bfloat16",
        },
        "prompts": {
            **{
                kind: template(kind)
                for kind in (
                    "setup",
                    "statement",
                    "ballot",
                    "correction",
                    "result",
                    "final_result",
                )
            },
            "reviewed": True,
            "review_version": "review/v1",
            "review_sha256": digest,
        },
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "max_new_tokens": 100,
        },
        "ballot_retry": {"max_corrections": 3},
        "statement_retry": {"max_corrections": 3},
        "runtime_retry": {
            "max_failures_per_execution": 3,
            "initial_backoff_ms": 1000,
            "multiplier": 2.0,
            "max_backoff_ms": 2000,
        },
        "master_seed": (1 << 64) - 1,
        "voter_count": 2,
        "credit_budget": 100,
        "sampler_policy": "balanced-matched/v1",
        "presentation_policy": "setup-once-ids-later/v1",
        "tie_policy": "uniform-seeded/v1",
        "action_format": "json-with-rationale/v1",
        "execution_class": "primary",
    }


class ConfigTest(unittest.TestCase):
    def test_complete_strict_config_accepts_full_uint64(self) -> None:
        config = MatchedSetConfigV1.model_validate_json(json.dumps(valid_config()))
        self.assertEqual(config.master_seed, (1 << 64) - 1)
        self.assertEqual(config.sample.artifact_path.as_posix(), "samples/one.json")

    def test_coercion_unknown_fields_bounds_and_retry_drift_are_rejected(self) -> None:
        mutations: tuple[tuple[tuple[str, ...], object], ...] = (
            (("master_seed",), True),
            (("voter_count",), "2"),
            (("sampling", "top_k"), 3.0),
            (("sampling", "top_p"), float("nan")),
            (("ballot_retry", "max_corrections"), 2),
            (("master_seed",), 1 << 64),
        )
        for path, value in mutations:
            data = copy.deepcopy(valid_config())
            target: dict[str, object] = data
            for component in path[:-1]:
                target = target[component]  # type: ignore[assignment]
            target[path[-1]] = value
            with self.subTest(path=path, value=value):
                with self.assertRaises(ValidationError):
                    MatchedSetConfigV1.model_validate(data)
        data = valid_config()
        data["unexpected"] = True
        with self.assertRaises(ValidationError):
            MatchedSetConfigV1.model_validate(data)


if __name__ == "__main__":
    unittest.main()
