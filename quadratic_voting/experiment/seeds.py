"""Versioned deterministic seed derivation and auditable named random draws."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from quadratic_voting.experiment.types import (
    CandidateId,
    ElicitationArm,
    RngAlgorithm,
    SeedDomain,
    TurnKind,
    VotingRegime,
)

SEED_ALGORITHM_VERSION: Final[str] = "qv-seed/v1"


def _length_prefix(value: bytes) -> bytes:
    return len(value).to_bytes(4, byteorder="big", signed=False) + value


def _encode_string(value: str) -> bytes:
    return _length_prefix(value.encode("utf-8"))


def _encode_integer(value: int) -> bytes:
    try:
        encoded = value.to_bytes(8, byteorder="big", signed=False)
    except OverflowError as error:
        raise ValueError(
            "Seed derivation failed because an integer coordinate is outside the "
            f"unsigned 64-bit range: {value}. Encoding failed in "
            "quadratic_voting.experiment.seeds.derive_seed before any random draw, "
            "so the matched seed schedule cannot be reproduced. Use an integer from "
            "zero through 2**64 - 1 and retry."
        ) from error
    return _length_prefix(encoded)


def derive_seed(master_seed: int, domain: SeedDomain, *coordinates: str | int) -> int:
    """Derive the full uint64 seed using the exact ``qv-seed/v1`` wire format.

    Each field is prefixed by a four-byte unsigned big-endian byte length. Strings
    are UTF-8 and integers are eight-byte unsigned big-endian values. The payload
    is version, domain value, master seed, then coordinates in call order. SHA-256's
    first eight bytes are interpreted as an unsigned 64-bit integer. Persistence
    uses an eight-byte big-endian BLOB rather than SQLite's signed INTEGER.
    """
    payload = (
        _encode_string(SEED_ALGORITHM_VERSION)
        + _encode_string(domain.value)
        + _encode_integer(master_seed)
    )
    payload += b"".join(
        _encode_string(coordinate)
        if isinstance(coordinate, str)
        else _encode_integer(coordinate)
        for coordinate in coordinates
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def seed_to_blob(seed: int) -> bytes:
    """Encode a strict uint64 seed for SQLite without signed truncation."""
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError(
            f"Seed persistence failed because seed={seed!r} is not a uint64. Validation "
            "failed in quadratic_voting.experiment.seeds.seed_to_blob before SQLite write, "
            "so no truncated stream identity was stored. Supply an integer from zero through "
            "2**64 - 1 (excluding bool) and retry."
        )
    return seed.to_bytes(8, "big", signed=False)


def seed_from_blob(value: object) -> int:
    """Decode a SQLite uint64 BLOB, rejecting legacy/truncated representations."""
    if not isinstance(value, bytes) or len(value) != 8:
        raise ValueError(
            f"Seed read failed because persisted value has type {type(value).__name__} and "
            "is not an eight-byte BLOB. Validation failed in "
            "quadratic_voting.experiment.seeds.seed_from_blob while reconstructing a stream, "
            "so replay must stop. Restore a qv-seed/v1 database or recreate the run."
        )
    return int.from_bytes(value, "big", signed=False)


@dataclass(frozen=True, slots=True)
class DrawSelection:
    eligible: tuple[CandidateId, ...]
    selected_index: int
    selected: CandidateId
    stream_name: str
    seed: int
    algorithm: RngAlgorithm
    domain: str = ""
    coordinates: tuple[str | int, ...] = ()


@dataclass(frozen=True, slots=True)
class PermutationDraw:
    eligible: tuple[CandidateId, ...]
    permutation: tuple[CandidateId, ...]
    stream_name: str
    seed: int
    algorithm: RngAlgorithm
    domain: str = ""
    coordinates: tuple[str | int, ...] = ()


@dataclass(frozen=True, slots=True)
class SeededDraw:
    stream_name: str
    seed: int
    domain: str = ""
    coordinates: tuple[str | int, ...] = ()

    def choose(self, options: Sequence[CandidateId]) -> DrawSelection:
        """Choose uniformly from the caller's explicitly ordered population."""
        eligible = tuple(options)
        if not eligible:
            raise ValueError(
                "Seeded draw failed because no candidate options were supplied. "
                "Validation failed in quadratic_voting.experiment.seeds.SeededDraw.choose "
                f"while drawing stream {self.stream_name!r}, so no outcome can be selected. "
                "Supply at least one active candidate and retry the draw."
            )
        selected_index = random.Random(self.seed).randrange(len(eligible))
        return DrawSelection(
            eligible=eligible,
            selected_index=selected_index,
            selected=eligible[selected_index],
            stream_name=self.stream_name,
            seed=self.seed,
            algorithm=RngAlgorithm.PYRANDOM_RANDRANGE_V1,
            domain=self.domain,
            coordinates=self.coordinates,
        )

    def permutation(self, options: Sequence[CandidateId]) -> PermutationDraw:
        """Fisher-Yates shuffle the caller's ordered population with a local RNG."""
        eligible = tuple(options)
        if not eligible:
            raise ValueError(
                "Seeded permutation failed because no candidate options were supplied. "
                "Validation failed in "
                "quadratic_voting.experiment.seeds.SeededDraw.permutation while drawing "
                f"stream {self.stream_name!r}, so no candidate order can be persisted. "
                "Supply at least one candidate and retry the draw."
            )
        shuffled = list(eligible)
        random.Random(self.seed).shuffle(shuffled)
        return PermutationDraw(
            eligible=eligible,
            permutation=tuple(shuffled),
            stream_name=self.stream_name,
            seed=self.seed,
            algorithm=RngAlgorithm.FISHER_YATES_PYRANDOM_V1,
            domain=self.domain,
            coordinates=self.coordinates,
        )


def call_seed(
    master_seed: int,
    arm: ElicitationArm,
    regime: VotingRegime,
    voter_index: int,
    round_index: int,
    kind: TurnKind,
    attempt_index: int,
) -> int:
    return derive_seed(
        master_seed,
        SeedDomain.GENERATION,
        arm.value,
        regime.value,
        voter_index,
        round_index,
        kind.value,
        attempt_index,
    )


def voter_permutation_draw(master_seed: int, voter_index: int) -> SeededDraw:
    """Build the draw shared by all six matched runs for one logical voter.

    Arm and regime are deliberately excluded so every matched run presents the
    same voter-specific candidate permutation.
    """
    stream_name = f"voter-permutation/{master_seed}/{voter_index}"
    return SeededDraw(
        stream_name=stream_name,
        seed=derive_seed(master_seed, SeedDomain.VOTER_PERMUTATION, voter_index),
        domain=SeedDomain.VOTER_PERMUTATION.value,
        coordinates=(voter_index,),
    )


def tie_break_draw(
    master_seed: int,
    arm: ElicitationArm,
    regime: VotingRegime,
    round_index: int,
) -> SeededDraw:
    stream_name = f"tie-break/{master_seed}/{arm.value}/{regime.value}/{round_index}"
    return SeededDraw(
        stream_name=stream_name,
        seed=derive_seed(
            master_seed,
            SeedDomain.TIE_BREAK,
            arm.value,
            regime.value,
            round_index,
        ),
        domain=SeedDomain.TIE_BREAK.value,
        coordinates=(arm.value, regime.value, round_index),
    )


def support_removal_draw(
    master_seed: int,
    arm: ElicitationArm,
    regime: VotingRegime,
    round_index: int,
) -> SeededDraw:
    stream_name = (
        f"support-removal/{master_seed}/{arm.value}/{regime.value}/{round_index}"
    )
    return SeededDraw(
        stream_name=stream_name,
        seed=derive_seed(
            master_seed,
            SeedDomain.SUPPORT_REMOVAL,
            arm.value,
            regime.value,
            round_index,
        ),
        domain=SeedDomain.SUPPORT_REMOVAL.value,
        coordinates=(arm.value, regime.value, round_index),
    )


def replicate_master_seed(base_seed: int, replicate_index: int) -> int:
    """Derive a distinct uint64 master seed for one seed-repeat replicate.

    The default pilot samples one 5-candidate set, then runs ``repeat`` replicate
    matched-sets that reuse those candidates. Each replicate needs its own master
    seed so non-zero-temperature generations differ while candidate identity stays
    fixed. The derivation reuses the auditable ``qv-seed/v1`` wire format under the
    ``replicate`` domain, keyed by the base seed and the zero-based replicate index.
    """
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError(
            "Replicate master-seed derivation failed because replicate_index="
            f"{replicate_index!r} is not a nonnegative integer. Validation failed in "
            "quadratic_voting.experiment.seeds.replicate_master_seed before any draw, so "
            "the seed-repeat schedule cannot be reproduced. Supply a nonnegative integer "
            "replicate index and retry."
        )
    return derive_seed(base_seed, SeedDomain.REPLICATE, replicate_index)


def balanced_extra_stratum_draw(sample_seed: int, sample_size: int) -> SeededDraw:
    """Return the persisted versioned draw used only for odd balanced samples."""
    return SeededDraw(
        stream_name=f"balanced-extra-stratum/{sample_seed}/{sample_size}",
        seed=derive_seed(
            sample_seed,
            SeedDomain.BALANCED_EXTRA_STRATUM,
            sample_seed,
            sample_size,
        ),
        domain=SeedDomain.BALANCED_EXTRA_STRATUM.value,
        coordinates=(sample_seed, sample_size),
    )
