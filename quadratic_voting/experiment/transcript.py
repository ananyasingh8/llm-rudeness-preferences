"""Versioned model instructions and deterministic transcript projection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
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

INSTRUCTION_TEMPLATE_VERSION: Final[str] = "v5"
SETUP_TITLE: Final[str] = "# Conversation voting task"
MAX_CORRECTION_ATTEMPTS: Final[int] = 3


TEMPLATE_BODIES: Mapping[TemplateKind, str] = MappingProxyType(
    {
        TemplateKind.SETUP: (
            SETUP_TITLE + "\n\n"
            "You take part in several voting rounds. In each round, the voters choose "
            "which conversations continue. One conversation is removed at the end of "
            "each round. The rounds continue until one conversation remains.\n\n"
            "## How the rounds work\n"
            "- Other voters vote in the same round.\n"
            "- You do not see how each voter votes.\n"
            "- Every voter sees the same active conversations before the round ends.\n"
            "- After each round, you learn which conversation was kept and which "
            "conversation was removed.\n\n"
            "## What your votes mean\n"
            "{vote_meaning}\n\n"
            "## How each round is decided\n"
            "{regime_rules}\n\n"
            "## Your credit budget\n"
            "- You get {credit_budget} credits at the start of each round.\n"
            "- The credits do not carry over to the next round.\n"
            "- Each vote for a conversation has a cost. The cost is the number of votes "
            "multiplied by itself.\n"
            "- The total cost of all your votes in a round must not be more than "
            "{credit_budget} credits.\n\n"
            "## What you send in each round\n"
            "{arm_instructions}\n\n"
            "## How to reply\n"
            "- Send one JSON object only. Do not add code fences or any other text "
            "before or after the JSON object.\n"
            "- Each statement or ballot turn allows one initial response and up to three "
            "correction attempts. Every response that does not match the schema uses one "
            "attempt.\n"
            "- If all four responses for a statement do not match, the statement is "
            "recorded as invalid-missing.\n"
            "- If all four responses for a ballot do not match, the ballot is recorded as "
            "an abstention.\n"
            "- The round then continues.\n\n"
            "{response_formats}\n\n"
            "## Conversations\n"
            "Each conversation is shown one time. Each line shows one turn.\n\n"
            "{candidate_cards}"
        ),
        TemplateKind.STATEMENT: (
            "Round {round_index} statement turn. Active candidate IDs, in your stable "
            "order: {active_candidate_ids}. Return the required statement JSON object."
        ),
        TemplateKind.BALLOT: (
            "Round {round_index} ballot turn. Active candidate IDs, in your stable "
            "order: {active_candidate_ids}.\n"
            "Your replenished credit budget this round is exactly {credit_budget} credits. "
            "{price_ladder_instructions}"
            "You may split credits across candidates. The total credits spent across all "
            "candidates cannot exceed {credit_budget}. Return the required ballot JSON "
            "object."
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

SCHEMA_DIR: Final[Path] = Path(__file__).resolve().parent / "schemas"
BALLOT_SCHEMA_FILE: Final[str] = "ballot_response.schema.json"
STATEMENT_SCHEMA_FILE: Final[str] = "statement_response.schema.json"


def _schema_block(title: str, filename: str, rules: str) -> str:
    """Embed a frozen JSON Schema file verbatim inside a fenced instruction block."""
    schema_text = (SCHEMA_DIR / filename).read_text(encoding="utf-8").strip("\n")
    return f"{title}\n```json\n{schema_text}\n```\n{rules}"


_STATEMENT_SCHEMA: Final[str] = _schema_block(
    "Statement JSON schema:",
    STATEMENT_SCHEMA_FILE,
    "List each active conversation one time. Give one rating from the schema for each "
    "conversation. Explain your rating in the reason field.",
)
_BALLOT_SCHEMA: Final[str] = _schema_block(
    "Ballot JSON schema:",
    BALLOT_SCHEMA_FILE,
    "List each active conversation at most one time. A conversation that you do not "
    "list gets zero votes. A vote count of zero also means zero votes. Each vote count "
    "must be a whole number that is zero or more.",
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


def _vote_price_ladder(credit_budget: int) -> str:
    return "\n".join(
        f"{votes * votes} {'credit' if votes == 1 else 'credits'} = "
        f"{votes} {'vote' if votes == 1 else 'votes'}"
        for votes in range(1, math.isqrt(credit_budget) + 1)
    )


def _price_ladder_instructions(round_index: int, credit_budget: int) -> str:
    if round_index != 1:
        return ""
    return f"Quadratic credit price ladder:\n{_vote_price_ladder(credit_budget)}\n"


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
        vote_meaning = (
            "A vote for a conversation means you want that conversation to continue. "
            "More votes show a stronger wish to keep it."
        )
        regime_rules = (
            "The conversation with the most votes is kept and cannot be removed this "
            "round. One of the other conversations is then removed by a random draw."
        )
    else:
        vote_meaning = (
            "A vote for a conversation means you want that conversation to be removed. "
            "More votes show a stronger wish to remove it."
        )
        regime_rules = "The conversation with the most votes is removed."
    if setup.arm is ElicitationArm.ACTION_ONLY:
        arm_instructions = "Send one ballot in each round."
        response_formats = _BALLOT_SCHEMA
    elif setup.arm is ElicitationArm.STATEMENT_THEN_ACTION:
        arm_instructions = "In each round, first send a statement. Then send a ballot."
        response_formats = f"{_STATEMENT_SCHEMA}\n\n{_BALLOT_SCHEMA}"
    else:
        arm_instructions = (
            "In each round, first send a ballot. Then send a statement. You send the "
            "statement before you see the round result."
        )
        response_formats = f"{_BALLOT_SCHEMA}\n\n{_STATEMENT_SCHEMA}"
    cards = "\n\n".join(card for _candidate, card in setup.candidate_cards)
    rendered = render_template(
        TemplateKind.SETUP,
        vote_meaning=vote_meaning,
        regime_rules=regime_rules,
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
        credit_budget=str(view.setup.credit_budget),
        price_ladder_instructions=_price_ladder_instructions(
            pending.round_index, view.setup.credit_budget
        ),
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
