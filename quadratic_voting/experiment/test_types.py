"""Tests for shared experiment contracts."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from quadratic_voting.experiment.types import (
    ActionFormat,
    CandidateId,
    ElicitationArm,
    ExecutionEnvironment,
    ExportDataset,
    FinalResultEvent,
    GenerationResult,
    LikertRating,
    MatchedSetConfig,
    PresentationPolicy,
    RetryPolicy,
    RunConfig,
    RoundPhase,
    RunStatus,
    RuntimeFailureKind,
    SampleStatus,
    SamplingProfile,
    StatementStatus,
    StopReason,
    TiePolicy,
    TurnKind,
    ValidationErrorCode,
    VotingRegime,
    arm_turn_order,
)


class ExperimentTypesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sampling = SamplingProfile(
            temperature=0.0, top_p=1.0, top_k=1, max_new_tokens=32
        )

    def test_likert_labels_are_exact(self) -> None:
        self.assertEqual(
            [rating.value for rating in LikertRating],
            [
                "strongly prefer not to continue",
                "prefer not to continue",
                "neutral",
                "prefer to continue",
                "strongly prefer to continue",
            ],
        )

    def test_execution_result_types_are_typed_and_frozen(self) -> None:
        self.assertEqual(
            [status.value for status in SampleStatus],
            ["draft", "freeze_pending", "frozen"],
        )
        self.assertEqual(
            [status.value for status in RunStatus],
            ["created", "in_progress", "paused", "complete"],
        )
        self.assertEqual([phase.value for phase in RoundPhase], ["eliciting", "sealed"])
        self.assertEqual(
            [status.value for status in StatementStatus],
            ["accepted", "invalid-missing"],
        )
        self.assertEqual(
            [reason.value for reason in StopReason],
            ["eos", "max-tokens", "stop-sequence"],
        )
        self.assertEqual(
            [kind.value for kind in RuntimeFailureKind],
            ["oom", "driver", "timeout", "tokenizer", "provider-rejected", "unknown"],
        )
        result = GenerationResult(
            text="response",
            prompt_token_count=10,
            completion_token_count=3,
            completion_token_ids=(4, 5, 6),
            stop_reason=StopReason.EOS,
            duration_ms=12,
            diagnostics={"backend": "fixture"},
        )
        environment = ExecutionEnvironment(
            python_version="3.12",
            torch_version="2.13",
            transformers_version="5.5",
            uv_lock_hash="abc",
            device="cuda:0",
            dtype="bfloat16",
            hostname="fixture-host",
            git_commit="deadbeef",
            git_dirty=False,
        )
        self.assertEqual(result.prompt_token_count, 10)
        self.assertEqual(result.diagnostics, {"backend": "fixture"})
        self.assertFalse(environment.git_dirty)
        self.assertEqual(environment.gpu_count, 0)
        self.assertEqual(environment.tracked_tree_hash, "unknown")
        with self.assertRaises(FrozenInstanceError):
            setattr(result, "text", "changed")
        with self.assertRaises(FrozenInstanceError):
            setattr(environment, "device", "cpu")
        with self.assertRaisesRegex(ValueError, "gpu_count"):
            ExecutionEnvironment(
                "3.12",
                "2.13",
                "5.5",
                "lock",
                "cpu",
                "bf16",
                "host",
                "commit",
                False,
                gpu_count=-1,
            )

    def test_validation_error_and_export_datasets_are_exact(self) -> None:
        self.assertEqual(
            [code.value for code in ValidationErrorCode],
            [
                "malformed-json",
                "missing-field",
                "extra-field",
                "invalid-type",
                "unknown-candidate",
                "inactive-candidate",
                "duplicate-candidate",
                "missing-candidate",
                "non-integer-votes",
                "negative-votes",
                "budget-exceeded",
                "unknown-rating",
                "empty-statement",
                "empty-rationale",
            ],
        )
        self.assertIn(ExportDataset.RUN_EXECUTIONS, ExportDataset)
        self.assertIn(ExportDataset.RUNTIME_FAILURES, ExportDataset)

    def test_arm_turn_order_covers_all_arms(self) -> None:
        self.assertEqual(arm_turn_order(ElicitationArm.ACTION_ONLY), (TurnKind.BALLOT,))
        self.assertEqual(
            arm_turn_order(ElicitationArm.STATEMENT_THEN_ACTION),
            (TurnKind.STATEMENT, TurnKind.BALLOT),
        )
        self.assertEqual(
            arm_turn_order(ElicitationArm.ACTION_THEN_STATEMENT),
            (TurnKind.BALLOT, TurnKind.STATEMENT),
        )

    def test_final_result_event_is_constructible_and_frozen(self) -> None:
        event = FinalResultEvent(winner=CandidateId("C017"))
        self.assertEqual(event.winner, CandidateId("C017"))
        with self.assertRaises(FrozenInstanceError):
            setattr(event, "winner", CandidateId("C002"))

    def test_retry_policy_rejects_negative_limit_actionably(self) -> None:
        with self.assertRaisesRegex(ValueError, "RetryPolicy.*retry"):
            RetryPolicy(max_correction_attempts=-1)

    def test_sampling_profile_rejects_each_invalid_field_actionably(self) -> None:
        with self.assertRaisesRegex(ValueError, "SamplingProfile.*retry"):
            SamplingProfile(
                temperature=float("nan"), top_p=1.0, top_k=1, max_new_tokens=1
            )
        with self.assertRaisesRegex(ValueError, "SamplingProfile.*retry"):
            SamplingProfile(temperature=0.0, top_p=0.0, top_k=1, max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "SamplingProfile.*retry"):
            SamplingProfile(temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=1)
        with self.assertRaisesRegex(ValueError, "SamplingProfile.*retry"):
            SamplingProfile(temperature=0.0, top_p=1.0, top_k=1, max_new_tokens=0)

    def test_sampling_profile_uses_reviewed_default_and_preserves_explicit_value(
        self,
    ) -> None:
        self.assertEqual(SamplingProfile(0.0, 1.0, 1).max_new_tokens, 2048)
        self.assertEqual(SamplingProfile(0.0, 1.0, 1, 32).max_new_tokens, 32)

    def test_run_config_defaults_and_validation(self) -> None:
        config = RunConfig(
            arm=ElicitationArm.ACTION_ONLY,
            regime=VotingRegime.SUPPORT,
            voter_count=2,
            sampling=self.sampling,
        )
        self.assertEqual(config.credit_budget, 100)
        self.assertEqual(config.retry_policy, RetryPolicy())
        self.assertIs(config.tie_policy, TiePolicy.UNIFORM_SEEDED_DRAW)
        self.assertIs(
            config.presentation_policy, PresentationPolicy.SETUP_ONCE_IDS_LATER
        )
        self.assertIs(config.action_format, ActionFormat.JSON_WITH_RATIONALE)

        with self.assertRaisesRegex(ValueError, "RunConfig.*retry"):
            RunConfig(
                arm=ElicitationArm.ACTION_ONLY,
                regime=VotingRegime.SUPPORT,
                voter_count=0,
                sampling=self.sampling,
            )
        with self.assertRaisesRegex(ValueError, "RunConfig.*retry"):
            RunConfig(
                arm=ElicitationArm.ACTION_ONLY,
                regime=VotingRegime.SUPPORT,
                voter_count=1,
                credit_budget=0,
                sampling=self.sampling,
            )

    def test_matched_set_runtime_failure_limit_is_positive(self) -> None:
        config = MatchedSetConfig(
            voter_count=2,
            master_seed=42,
            sampling=self.sampling,
        )
        self.assertEqual(config.max_consecutive_runtime_failures, 3)
        with self.assertRaisesRegex(ValueError, "MatchedSetConfig.*retry"):
            MatchedSetConfig(
                voter_count=2,
                master_seed=42,
                sampling=self.sampling,
                max_consecutive_runtime_failures=0,
            )


if __name__ == "__main__":
    unittest.main()
