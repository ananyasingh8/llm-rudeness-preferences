"""Typed actionable failures at experiment persistence boundaries."""

from __future__ import annotations


class ExperimentError(RuntimeError):
    """Base class for failures that must stop or fork an experiment."""


class DefinitionDriftError(ExperimentError):
    """An immutable scientific definition differs from the persisted definition."""


class WriterLockError(ExperimentError):
    """Another process owns the kernel-enforced writer boundary."""


class FreezeMismatchError(ExperimentError):
    """A final sample artifact conflicts with its durable freeze intent."""


class DirtyPrimaryTreeError(ExperimentError):
    """Primary execution was attempted from a dirty source tree."""
