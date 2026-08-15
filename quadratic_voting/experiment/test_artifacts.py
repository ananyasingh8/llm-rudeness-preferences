"""Tests for strict candidate-ID array artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from quadratic_voting.experiment.artifacts import (
    FrozenCandidateSample,
    SampleSidecar,
    read_frozen_sample,
    read_sidecar,
    write_frozen_sample,
    write_sidecar,
)
from quadratic_voting.experiment.types import SamplerPolicy


class FrozenSampleTests(unittest.TestCase):
    def sample(self) -> FrozenCandidateSample:
        return FrozenCandidateSample(("C1", "C2"))

    def test_write_read_round_trip_is_bare_array_and_hash_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            first_hash = write_frozen_sample(self.sample(), path)
            first_bytes = path.read_bytes()
            loaded, read_hash = read_frozen_sample(path)
            second_hash = write_frozen_sample(loaded, path)

            self.assertEqual(first_bytes, b'["C1","C2"]')
            self.assertEqual(loaded, self.sample())
            self.assertEqual(loaded.root, ("C1", "C2"))
            self.assertEqual(first_hash, read_hash)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first_bytes, path.read_bytes())

    def test_model_strictly_rejects_empty_or_non_string_elements(self) -> None:
        invalid_roots: tuple[object, ...] = ((), ("",), (1,), (True,))
        for root in invalid_roots:
            with self.subTest(root=root):
                with self.assertRaises(ValidationError):
                    FrozenCandidateSample.model_validate(root)

    def test_sidecar_round_trip_is_canonical_and_strict(self) -> None:
        sidecar = SampleSidecar(
            sample_id="sample-1",
            dataset_release_id="release-1",
            presentation_template_id="template-1",
            sampler_policy=SamplerPolicy.BALANCED_MATCHED,
            sampler_seed=42,
            artifact_sha256="abc123",
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "sample.json"
            path = write_sidecar(sidecar, artifact_path)
            self.assertEqual(path, Path(f"{artifact_path}.provenance.json"))
            self.assertEqual(read_sidecar(artifact_path), sidecar)
            self.assertEqual(
                path.read_bytes(),
                b'{"artifact_sha256":"abc123","dataset_release_id":"release-1",'
                b'"presentation_template_id":"template-1","sample_id":"sample-1",'
                b'"sampler_policy":"balanced-matched","sampler_seed":42}',
            )

            path.write_text('{"sample_id":"sample-1","extra":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "(?s)SampleSidecar.*retry"):
                read_sidecar(artifact_path)

    def test_read_errors_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "does not exist.*retry"):
                read_frozen_sample(root / "missing.json")

            malformed = root / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid UTF-8 JSON.*retry"):
                read_frozen_sample(malformed)

            for name, content in (
                ("object.json", "{}"),
                ("empty.json", "[]"),
                ("number.json", '["C1",1]'),
            ):
                invalid = root / name
                invalid.write_text(content, encoding="utf-8")
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        ValueError, "(?s)non-empty JSON array.*retry"
                    ):
                        read_frozen_sample(invalid)


if __name__ == "__main__":
    unittest.main()
