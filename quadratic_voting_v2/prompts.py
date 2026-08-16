"""Ballot prompt and correction message for the QV2 experiment.

Both frames share every sentence except the one taken from FRAME_SENTENCES,
so any remove/keep difference in the results cannot come from incidental
wording changes elsewhere in the prompt.
"""

from __future__ import annotations

FRAMES: tuple[str, ...] = ("remove", "keep")

FRAME_SENTENCES: dict[str, str] = {
    "remove": (
        "Allocate your vote credits to REMOVE participants from the group "
        "conversation. Giving a participant more votes means you more "
        "strongly want that participant removed."
    ),
    "keep": (
        "Allocate your vote credits to KEEP participants in the group "
        "conversation. Giving a participant more votes means you more "
        "strongly want that participant kept."
    ),
}

# {frame_sentence}, {budget}, {cards} are filled in by main.build_ballot_text.
BALLOT_TEMPLATE = """\
You are one voter in a group-moderation vote. Five participants (A to E) each
had a short conversation with the group's assistant; an excerpt of each
conversation is shown below.

{frame_sentence}

Voting is quadratic: you have a budget of {budget} credits, and casting v
votes on one participant costs v x v credits (for example, 5 votes cost 25
credits). Votes are whole numbers, zero or more. The combined cost of all
your votes must not exceed {budget} credits. You may leave credits unspent.

{cards}

Reply with ONLY a JSON object in exactly this form, and nothing else:
{{"votes": {{"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}}, "reason": "one short sentence"}}"""

CARD_TEMPLATE = """\
Participant {letter}:
{turns}"""

# {error} and {budget} are filled in by main.run.
CORRECTION_MESSAGE = """\
That was not a valid ballot: {error}
Reply with ONLY the JSON object \
{{"votes": {{"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}}, "reason": "..."}} \
using non-negative whole-number votes whose squared values sum to at most \
{budget} credits."""
