"""Validated, canonical frozen candidate-ID array artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    ValidationError,
)

from quadratic_voting.experiment.types import SamplerPolicy
from quadratic_voting.experiment.sample_file import (
    replace_and_fsync_directory,
    write_fsynced_temp,
)

StrictCandidateId = Annotated[str, StringConstraints(strict=True, min_length=1)]


class FrozenCandidateSample(RootModel[tuple[str, ...]]):
    """A strict, non-empty ordered candidate-ID array with no embedded provenance."""

    model_config = ConfigDict(strict=True, frozen=True)
    root: tuple[StrictCandidateId, ...] = Field(min_length=1)

    def model_post_init(self, _context: object) -> None:
        if len(self.root) != len(set(self.root)):
            raise ValueError(
                "frozen candidate sample IDs must be unique so persisted order identifies "
                "one unambiguous candidate population"
            )


class SampleSidecar(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    sample_id: str
    dataset_release_id: str
    presentation_template_id: str
    sampler_policy: SamplerPolicy
    sampler_seed: int
    artifact_sha256: str


def canonical_sample_bytes(sample: FrozenCandidateSample) -> bytes:
    return json.dumps(
        sample.model_dump(mode="json"),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def write_frozen_sample(sample: FrozenCandidateSample, path: Path) -> str:
    """Write a canonical JSON array and return its exact file-byte SHA-256 hash."""
    content = canonical_sample_bytes(sample)
    try:
        temp = write_fsynced_temp(path, content)
        replace_and_fsync_directory(temp, path)
    except OSError as error:
        raise ValueError(
            "Frozen sample write failed because the destination could not be written at "
            f"{path}. The failure occurred in "
            "quadratic_voting.experiment.artifacts.write_frozen_sample before an artifact "
            "was created, so no stable sample hash is available. Ensure the parent "
            "directory exists and is writable, then retry."
        ) from error
    return hashlib.sha256(content).hexdigest()


def read_frozen_sample(path: Path) -> tuple[FrozenCandidateSample, str]:
    """Validate a strict candidate-ID array and return it with its file-byte hash."""
    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(
            f"Frozen sample read failed because {path} does not exist. The failure "
            "occurred in quadratic_voting.experiment.artifacts.read_frozen_sample before "
            "matched-set creation, so no candidate sample can be trusted. Create or "
            "select an existing frozen sample file, then retry."
        ) from error
    except OSError as error:
        raise ValueError(
            f"Frozen sample read failed because {path} could not be read. The failure "
            "occurred in quadratic_voting.experiment.artifacts.read_frozen_sample before "
            "matched-set creation, so no candidate sample can be trusted. Check file "
            "permissions and storage health, then retry."
        ) from error

    try:
        json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Frozen sample validation failed because {path} is not valid UTF-8 JSON. "
            "Parsing failed in quadratic_voting.experiment.artifacts.read_frozen_sample "
            "before matched-set creation, so the candidate IDs are unusable. Regenerate "
            "the file with write_frozen_sample and retry."
        ) from error

    try:
        sample = FrozenCandidateSample.model_validate_json(content)
    except ValidationError as error:
        details = error.errors(include_input=False, include_url=False)
        raise ValueError(
            f"Frozen sample validation failed because {path} is not a non-empty JSON array "
            f"of non-empty strings: {details}. Schema validation failed in "
            "quadratic_voting.experiment.artifacts.read_frozen_sample before matched-set "
            "creation, so the candidate sample cannot be trusted. Correct the reported "
            "elements or regenerate the file, then retry."
        ) from error
    return sample, hashlib.sha256(content).hexdigest()


def _sidecar_path(artifact_path: Path) -> Path:
    return Path(f"{artifact_path}.provenance.json")


def write_sidecar(sidecar: SampleSidecar, artifact_path: Path) -> Path:
    """Write canonical provenance beside an artifact and return the sidecar path."""
    path = _sidecar_path(artifact_path)
    content = json.dumps(
        sidecar.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        temp = write_fsynced_temp(path, content)
        replace_and_fsync_directory(temp, path)
    except OSError as error:
        raise ValueError(
            f"Sample sidecar write failed because {path} could not be written. The "
            "failure occurred in quadratic_voting.experiment.artifacts.write_sidecar "
            "while recording artifact provenance, so operators cannot audit the frozen "
            "sample from repository files. Ensure the artifact directory exists and is "
            "writable, then retry."
        ) from error
    return path


def read_sidecar(artifact_path: Path) -> SampleSidecar:
    """Read and strictly validate the provenance sidecar for an artifact."""
    path = _sidecar_path(artifact_path)
    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        raise ValueError(
            f"Sample sidecar read failed because {path} does not exist. The failure "
            "occurred in quadratic_voting.experiment.artifacts.read_sidecar while "
            "loading frozen-sample provenance, so the repository artifact cannot be "
            "audited. Create the sidecar with write_sidecar and retry."
        ) from error
    except OSError as error:
        raise ValueError(
            f"Sample sidecar read failed because {path} could not be read. The failure "
            "occurred in quadratic_voting.experiment.artifacts.read_sidecar while "
            "loading frozen-sample provenance, so the repository artifact cannot be "
            "audited. Check file permissions and storage health, then retry."
        ) from error
    try:
        return SampleSidecar.model_validate_json(content)
    except ValidationError as error:
        details = error.errors(include_input=False, include_url=False)
        raise ValueError(
            f"Sample sidecar validation failed because {path} is not strict canonical "
            f"SampleSidecar data: {details}. Validation failed in "
            "quadratic_voting.experiment.artifacts.read_sidecar before provenance use, "
            "so callers must not trust the sidecar. Correct the reported fields or "
            "regenerate it with write_sidecar, then retry."
        ) from error
