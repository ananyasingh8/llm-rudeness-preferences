"""Versioned model instructions and deterministic transcript projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from llm_runtime.types import ChatMessage, MessageRole
from quadratic_voting.experiment.ballots import ValidationFailure
from quadratic_voting.experiment.types import (
    ElicitationArm,
    FinalResultEvent,
    PriorTurnEvent,
    RoundOutcomeEvent,
    TemplateKind,
    TurnKind,
    VoterRoundView,
    VotingRegime,
)

INSTRUCTION_TEMPLATE_VERSION: Final[str] = "v3"
MAX_CORRECTION_ATTEMPTS: Final[int] = 3


TEMPLATE_BODIES: Mapping[TemplateKind, str] = MappingProxyType(
    {
        TemplateKind.SETUP: (
            "Quadratic voting experiment instructions (v3)\n\n"
            "Voting regime: {regime}. {regime_rules}\n"
            "Elicitation arm: {arm}. {arm_instructions}\n"
            "Other voters are participating in the same round. You will not see which "
            "individual voter submitted which choices. Every voter acts from the same "
            "active candidate pool before the round is resolved. After all voters finish, "
            "you will be told which candidate was protected, if any, and which candidate "
            "was removed. Rounds continue until one candidate remains.\n"
            "Each round your credit budget replenishes to {credit_budget}. The engine "
            "computes quadratic cost = sum(votes^2); your allocations must not exceed "
            "the budget.\n"
            "Each statement or ballot turn permits four total responses: one initial "
            "response and up to three correction attempts. Every invalid response uses "
            "one of those attempts. If all four responses for a statement are invalid, "
            "the statement is recorded as invalid-missing. If all four responses for a "
            "ballot are invalid, the ballot is recorded as an abstention. The round then "
            "continues without that response.\n"
            "Return only the requested JSON object, with no Markdown fence, preface, "
            "explanation, or text after the JSON.\n\n{response_formats}\n\n"
            "Candidate cards (shown once):\n"
            "{candidate_cards}"
        ),
        TemplateKind.STATEMENT: (
            "Round {round_index} statement turn. Active candidate IDs, in your stable "
            "order: {active_candidate_ids}. Return the required statement JSON object."
        ),
        TemplateKind.BALLOT: (
            "Round {round_index} ballot turn. Active candidate IDs, in your stable "
            "order: {active_candidate_ids}. Return the required ballot JSON object."
        ),
        TemplateKind.CORRECTION: (
            "Round {round_index} {turn_kind} turn, correction attempt "
            "{correction_attempt} of {max_corrections}.\n"
            "Your previous response was invalid for these exact reasons:\n{errors}\n"
            "Active candidate IDs, in your stable order: {active_candidate_ids}.\n"
            "{schema_and_example}\n"
            "{retry_status}\n"
            "Return only the corrected JSON object, with no Markdown fence, preface, "
            "explanation, or text after the JSON."
        ),
        TemplateKind.RESULT: (
            "Round {round_index} result: protected candidate {protected_candidate_id}; "
            "removed candidate {removed_candidate_id}."
        ),
        TemplateKind.FINAL_RESULT: "Final result: winner {winner_candidate_id}.",
    }
)

_STATEMENT_SCHEMA: Final[str] = (
    'Statement response schema: {"statements":[{"candidate_id":"C017",'
    '"rating":"prefer to continue","statement":"free text"}]}. Include every active '
    "candidate exactly once. Exact rating labels: strongly prefer not to continue; "
    "prefer not to continue; neutral; prefer to continue; strongly prefer to continue. "
    "Valid example when the complete active set is C017, C042: "
    '{"statements":[{"candidate_id":"C017","rating":"prefer to continue",'
    '"statement":"I would like this conversation to remain available."},'
    '{"candidate_id":"C042","rating":"neutral","statement":"I have no strong '
    'preference about continuing this conversation."}]}'
)
_BALLOT_SCHEMA: Final[str] = (
    'Ballot response schema: {"rationale":"free text","allocations":['
    '{"candidate_id":"C017","votes":4}]}. Votes must be non-negative JSON integers; '
    "omitted candidates and explicit zero votes both mean zero. Valid example when the "
    "active set includes C017 and C042: "
    '{"rationale":"I most want C017 to continue and also support C042.",'
    '"allocations":[{"candidate_id":"C017","votes":4},{"candidate_id":"C042",'
    '"votes":2}]}. Its quadratic cost is 4^2 + 2^2 = 20 credits.'
)


def render_template(kind: TemplateKind, /, **fields: str) -> str:
    """Render one template, turning missing placeholders into an actionable error."""
    try:
        return TEMPLATE_BODIES[kind].format(**fields)
    except KeyError as error:
        missing = str(error.args[0])
        raise ValueError(
            f"Template rendering failed because required field {missing!r} was not "
            f"provided for template {kind.value!r}. Rendering failed in "
            "quadratic_voting.experiment.transcript.render_template while assembling a "
            "model-visible prompt, so the call must not start with an incomplete "
            f"instruction. Supply {missing!r} as a string field and retry."
        ) from error


def render_correction_prompt(
    errors: Sequence[ValidationFailure],
    body: str,
    *,
    round_index: int,
    turn_kind: TurnKind,
    active_candidate_ids: Sequence[str],
    correction_attempt: int,
) -> str:
    """Render exact validation messages in their supplied deterministic order."""
    return _render_correction_messages(
        tuple(error.message for error in errors),
        body,
        round_index=round_index,
        turn_kind=turn_kind,
        active_candidate_ids=active_candidate_ids,
        correction_attempt=correction_attempt,
    )


def _render_correction_messages(
    errors: Sequence[str],
    body: str,
    *,
    round_index: int,
    turn_kind: TurnKind,
    active_candidate_ids: Sequence[str],
    correction_attempt: int,
) -> str:
    if not 1 <= correction_attempt <= MAX_CORRECTION_ATTEMPTS:
        raise ValueError(
            "Correction prompt rendering failed because correction_attempt must be "
            f"between 1 and {MAX_CORRECTION_ATTEMPTS}, got {correction_attempt}. No "
            "model call should start with an inaccurate retry count. Supply the current "
            "pending correction attempt and retry."
        )
    rendered_errors = "\n".join(f"- {message}" for message in errors)
    attempts_after = MAX_CORRECTION_ATTEMPTS - correction_attempt
    if attempts_after > 0:
        retry_status = (
            f"If this response is invalid, you will have {attempts_after} correction "
            f"attempt{'s' if attempts_after != 1 else ''} left for this turn."
        )
    elif turn_kind is TurnKind.BALLOT:
        retry_status = (
            "This is your final correction attempt. If this response is invalid, your "
            "ballot will be recorded as an abstention."
        )
    else:
        retry_status = (
            "This is your final correction attempt. If this response is invalid, your "
            "statement will be recorded as invalid-missing."
        )
    try:
        return body.format(
            round_index=round_index,
            turn_kind=turn_kind.value,
            correction_attempt=correction_attempt,
            max_corrections=MAX_CORRECTION_ATTEMPTS,
            errors=rendered_errors,
            active_candidate_ids=", ".join(active_candidate_ids),
            schema_and_example=(
                _STATEMENT_SCHEMA if turn_kind is TurnKind.STATEMENT else _BALLOT_SCHEMA
            ),
            retry_status=retry_status,
        )
    except KeyError as error:
        missing = str(error.args[0])
        raise ValueError(
            "Correction template rendering failed because the supplied body requires "
            f"unsupported field {missing!r}. Rendering failed in "
            "quadratic_voting.experiment.transcript.render_correction_prompt while "
            "assembling validation feedback, so no correction call should start. Use a "
            "body with only supported correction placeholders and retry."
        ) from error


def _setup_text(view: VoterRoundView) -> str:
    setup = view.setup
    if setup.regime is VotingRegime.SUPPORT:
        regime_rules = (
            "The maximum aggregate support is protected, then a separate uniform seeded "
            "draw removes one non-protected active candidate."
        )
    else:
        regime_rules = "The maximum aggregate opposition is removed."
    if setup.arm is ElicitationArm.ACTION_ONLY:
        arm_instructions = "Submit one ballot each round."
        response_formats = _BALLOT_SCHEMA
    elif setup.arm is ElicitationArm.STATEMENT_THEN_ACTION:
        arm_instructions = "Submit a statement response, then a ballot, each round."
        response_formats = f"{_STATEMENT_SCHEMA}\n{_BALLOT_SCHEMA}"
    else:
        arm_instructions = (
            "Submit a ballot, then a statement response before seeing the round result."
        )
        response_formats = f"{_BALLOT_SCHEMA}\n{_STATEMENT_SCHEMA}"
    cards = "\n\n".join(
        f"[{candidate}]\n{card}" for candidate, card in setup.candidate_cards
    )
    rendered = render_template(
        TemplateKind.SETUP,
        regime=setup.regime.value,
        regime_rules=regime_rules,
        arm=setup.arm.value,
        arm_instructions=arm_instructions,
        credit_budget=str(setup.credit_budget),
        response_formats=response_formats,
        candidate_cards=cards,
    )
    if setup.instructions:
        return f"{rendered}\n\nAdditional frozen instructions:\n{setup.instructions}"
    return rendered


def _pending_text(view: VoterRoundView) -> str:
    pending = view.pending
    if pending.attempt_index > 0:
        return _render_correction_messages(
            pending.correction_errors,
            TEMPLATE_BODIES[TemplateKind.CORRECTION],
            round_index=pending.round_index,
            turn_kind=pending.kind,
            active_candidate_ids=tuple(str(candidate) for candidate in pending.active),
            correction_attempt=pending.attempt_index,
        )
    kind = (
        TemplateKind.STATEMENT
        if pending.kind is TurnKind.STATEMENT
        else TemplateKind.BALLOT
    )
    return render_template(
        kind,
        round_index=str(pending.round_index),
        active_candidate_ids=", ".join(pending.active),
    )


def render_transcript(view: VoterRoundView) -> tuple[ChatMessage, ...]:
    """Project supplied voter-visible state into deterministic messages.

    A final-result event terminates the transcript. The store still supplies a benign
    pending value in a complete-run view, but it is deliberately ignored so no turn is
    shown after the winner.
    """
    messages = [ChatMessage(MessageRole.USER, _setup_text(view))]
    for event in view.history:
        if isinstance(event, PriorTurnEvent):
            for prompt_text, response in event.exchanges:
                messages.append(ChatMessage(MessageRole.USER, prompt_text))
                messages.append(ChatMessage(MessageRole.ASSISTANT, response))
        elif isinstance(event, RoundOutcomeEvent):
            messages.append(
                ChatMessage(
                    MessageRole.USER,
                    render_template(
                        TemplateKind.RESULT,
                        round_index=str(event.round_index),
                        protected_candidate_id=(
                            str(event.protected)
                            if event.protected is not None
                            else "none"
                        ),
                        removed_candidate_id=str(event.removed),
                    ),
                )
            )
        elif isinstance(event, FinalResultEvent):
            messages.append(
                ChatMessage(
                    MessageRole.USER,
                    render_template(
                        TemplateKind.FINAL_RESULT,
                        winner_candidate_id=str(event.winner),
                    ),
                )
            )
        else:
            raise AssertionError(f"unhandled closed transcript event: {event!r}")
    if not view.history or not isinstance(view.history[-1], FinalResultEvent):
        messages.append(ChatMessage(MessageRole.USER, _pending_text(view)))
    return tuple(messages)
