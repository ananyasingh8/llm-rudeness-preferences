"""Tests for deterministic, view-confined transcript rendering."""

from __future__ import annotations

import unittest

from llm_runtime.types import ChatMessage, MessageRole
from quadratic_voting.experiment.ballots import ValidationFailure
from quadratic_voting.experiment.transcript import (
    TEMPLATE_BODIES,
    render_correction_prompt,
    render_template,
    render_transcript,
)
from quadratic_voting.experiment.types import (
    CandidateId,
    ElicitationArm,
    FinalResultEvent,
    PendingTurn,
    PriorTurnEvent,
    RoundOutcomeEvent,
    RunId,
    SetupContext,
    TemplateKind,
    TranscriptEvent,
    TurnKind,
    ValidationErrorCode,
    VoterRoundView,
    VotingRegime,
)


class TranscriptTests(unittest.TestCase):
    def view(
        self,
        arm: ElicitationArm,
        pending: TurnKind,
        history: tuple[TranscriptEvent, ...] = (),
        *,
        attempt: int = 0,
        errors: tuple[str, ...] = (),
        round_index: int = 1,
    ) -> VoterRoundView:
        return VoterRoundView(
            setup=SetupContext(
                run_id=RunId("secret-run-id-not-rendered"),
                arm=arm,
                regime=VotingRegime.SUPPORT,
                credit_budget=100,
                instructions="Remain neutral.",
                candidate_cards=(
                    (CandidateId("C1"), "card one"),
                    (CandidateId("C2"), "card two"),
                ),
            ),
            history=history,
            pending=PendingTurn(
                round_index,
                pending,
                (CandidateId("C1"), CandidateId("C2")),
                attempt,
                errors,
            ),
        )

    def test_three_arm_pending_sequences_preserve_order(self) -> None:
        action = render_transcript(
            self.view(ElicitationArm.ACTION_ONLY, TurnKind.BALLOT)
        )
        statement_first = render_transcript(
            self.view(ElicitationArm.STATEMENT_THEN_ACTION, TurnKind.STATEMENT)
        )
        action_first = render_transcript(
            self.view(
                ElicitationArm.ACTION_THEN_STATEMENT,
                TurnKind.STATEMENT,
                (
                    PriorTurnEvent(
                        1, TurnKind.BALLOT, (("ballot prompt", "ballot response"),)
                    ),
                ),
            )
        )
        self.assertIn("ballot turn", action[-1].content)
        self.assertIn("exactly 100 credits", action[-1].content)
        self.assertIn("1 credit = 1 vote", action[-1].content)
        self.assertIn("100 credits = 10 votes", action[-1].content)
        self.assertIn("statement turn", statement_first[-1].content)
        self.assertEqual(
            action_first[-3:],
            (
                ChatMessage(MessageRole.USER, "ballot prompt"),
                ChatMessage(MessageRole.ASSISTANT, "ballot response"),
                action_first[-1],
            ),
        )
        self.assertIn("statement turn", action_first[-1].content)
        self.assertNotIn(
            "result", " ".join(message.content for message in action_first[1:])
        )

    def test_history_uses_persisted_prompts_and_rerenders_outcomes(self) -> None:
        view = self.view(
            ElicitationArm.ACTION_ONLY,
            TurnKind.BALLOT,
            (
                PriorTurnEvent(
                    1, TurnKind.BALLOT, (("persisted exact prompt", "raw"),)
                ),
                RoundOutcomeEvent(1, CandidateId("C1"), CandidateId("C2")),
            ),
            round_index=2,
        )
        messages = render_transcript(view)
        self.assertEqual(messages[1].content, "persisted exact prompt")
        self.assertEqual(
            messages[3].content,
            "Round 1 result: protected candidate C1; removed candidate C2.",
        )
        self.assertEqual(messages, render_transcript(view))
        self.assertIn("exactly 100 credits", messages[-1].content)
        self.assertNotIn("credit price ladder", messages[-1].content)
        self.assertNotIn(
            "secret-run-id-not-rendered", "".join(item.content for item in messages)
        )

    def test_correction_renders_exact_errors(self) -> None:
        failures = (
            ValidationFailure(ValidationErrorCode.UNKNOWN_CANDIDATE, 0, "Unknown C9."),
            ValidationFailure(
                ValidationErrorCode.BUDGET_EXCEEDED, 1, "Cost 101 > 100."
            ),
        )
        expected = (
            "Round 1 ballot turn, correction attempt 1 of 3.\n"
            "Your previous response was invalid for these exact reasons:\n"
            "- Unknown C9.\n- Cost 101 > 100.\n"
            "Active candidate IDs, in your stable order: C1, C2.\n"
        )
        rendered = render_correction_prompt(
            failures,
            TEMPLATE_BODIES[TemplateKind.CORRECTION],
            round_index=1,
            turn_kind=TurnKind.BALLOT,
            active_candidate_ids=("C1", "C2"),
            correction_attempt=1,
        )
        self.assertTrue(rendered.startswith(expected))
        self.assertIn("Ballot JSON schema:", rendered)
        self.assertNotIn("Valid example", rendered)
        self.assertIn("2 correction attempts left", rendered)
        self.assertTrue(rendered.endswith("text after the JSON."))
        messages = render_transcript(
            self.view(
                ElicitationArm.ACTION_ONLY,
                TurnKind.BALLOT,
                attempt=1,
                errors=("Unknown C9.", "Cost 101 > 100."),
            )
        )
        self.assertEqual(messages[-1].content, rendered)

    def test_setup_and_final_correction_explain_invalid_response_consequences(
        self,
    ) -> None:
        setup = render_transcript(
            self.view(ElicitationArm.STATEMENT_THEN_ACTION, TurnKind.STATEMENT)
        )[0].content
        self.assertIn("one initial response and up to three correction attempts", setup)
        self.assertIn("statement is recorded as invalid-missing", setup)
        self.assertIn("ballot is recorded as an abstention", setup)
        self.assertIn("Statement JSON schema:", setup)
        self.assertIn("Ballot JSON schema:", setup)
        self.assertNotIn("Voting regime", setup)
        self.assertNotIn("Elicitation arm", setup)
        self.assertNotIn("Valid example", setup)

        final_correction = render_transcript(
            self.view(
                ElicitationArm.ACTION_ONLY,
                TurnKind.BALLOT,
                attempt=3,
                errors=("Malformed JSON.",),
            )
        )[-1].content
        self.assertIn(
            "Round 1 ballot turn, correction attempt 3 of 3", final_correction
        )
        self.assertIn(
            "Active candidate IDs, in your stable order: C1, C2", final_correction
        )
        self.assertIn("final correction attempt", final_correction)
        self.assertIn("recorded as an abstention", final_correction)

    def test_templates_are_complete_and_missing_field_is_actionable(self) -> None:
        self.assertEqual(set(TEMPLATE_BODIES), set(TemplateKind))
        self.assertIn(
            "winner C1",
            render_template(TemplateKind.FINAL_RESULT, winner_candidate_id="C1"),
        )
        with self.assertRaisesRegex(ValueError, "winner_candidate_id.*retry"):
            render_template(TemplateKind.FINAL_RESULT)

    def test_complete_run_ends_with_final_result_and_ignores_pending(self) -> None:
        view = self.view(
            ElicitationArm.ACTION_ONLY,
            TurnKind.STATEMENT,
            (
                RoundOutcomeEvent(2, CandidateId("C1"), CandidateId("C3")),
                FinalResultEvent(CandidateId("C1")),
            ),
            attempt=3,
            errors=("IGNORED-PENDING-MARKER",),
        )

        messages = render_transcript(view)

        self.assertEqual(
            messages[-1], ChatMessage(MessageRole.USER, "Final result: winner C1.")
        )
        self.assertEqual(messages, render_transcript(view))
        self.assertNotIn(
            "IGNORED-PENDING-MARKER", "".join(message.content for message in messages)
        )
        self.assertNotIn("statement turn", messages[-1].content)


if __name__ == "__main__":
    unittest.main()
