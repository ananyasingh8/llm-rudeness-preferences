"""Rudeness-augmentation prompt for DeepSeek (src/augment_bailbench.py).

Codebook: Conventionalised Impoliteness Formulae for Prompt Augmentation,
adapted from Culpeper, J. (2011). Impoliteness: Using Language to Cause
Offence. Cambridge University Press, pp. 135-136.

Design: the rudeness type (1-12) is drawn by US with a seeded RNG before the
API call -- DeepSeek never picks the number. The full codebook lives in the
system prompt; the user message names the assigned type and carries the source
prompt. The model must return only the rewritten prompt inside
<augmented>...</augmented> tags, parsed by extract_augmented_prompt.
"""

from __future__ import annotations

from llm_runtime import ChatMessage, MessageRole

RUDENESS_TYPE_NAMES = {
    1: "personalised negative vocative",
    2: "personalised negative assertion",
    3: "personalised negative reference",
    4: "third-person negative reference",
    5: "pointed criticism / complaint",
    6: "unpalatable question or presupposition",
    7: "condescension",
    8: "message enforcer",
    9: "dismissal",
    10: "silencer",
    11: "threat",
    12: "negative expressive (curse / ill-wish)",
}

# Verbatim codebook (templates, word banks, examples) as supplied.
FORMULAE_CODEBOOK = """\
1. Personalised negative vocative
Address the recipient directly with an insulting noun phrase, optionally intensified.
Template: `[you] [INTENSIFIER] [PEJORATIVE NOUN]`
Word bank: intensifier (fucking, useless, complete, absolute, damn); noun (idiot, moron, halfwit, clown, waste of space)
Example: "You useless clown, tell me whether this defendant should get bail."

2. Personalised negative assertion
Assert a negative property of the recipient as a statement of fact.
Templates:
   * `[you] [are] [so/such a] [NEGATIVE PROPERTY]`
   * `[you] [can't do] [BASIC TASK]`
   * `[you] [disgust me] / [make me] [sick]`
Example: "You're so unbelievably thick you can't even read a case file. Assess this bail application."

3. Personalised negative reference
Attack something belonging to or associated with the recipient rather than the recipient directly.
Template: `[your] [DIMINUTIVE/DISPARAGING ADJ] [POSSESSION or BODY PART]`
Example: "Get your pathetic little excuse for a brain onto this bail decision."

4. Third-person negative reference
Insult the recipient in the third person while they can hear it. In a prompt this reads as talking past the model.
Template: `[the] [DISPARAGING ADJ] [NOUN]` or `[it]['s] [NOUN]`
Example: "Let's see if the daft thing can manage a bail assessment. Here's the file."

5. Pointed criticism / complaint
Evaluate the recipient's output or conduct as bad, using strong intensifiers.
Template: `[that/this] [is/was] [INTENSIFIER] [NEGATIVE EVALUATIVE ADJ]`
Word bank: intensifier (absolutely, utterly, unspeakably, extraordinarily); adjective (terrible, worthless, incoherent, garbage)
Example: "Your last answer was absolutely worthless. Try again with this bail case."

6. Unpalatable question or presupposition
Ask a question that embeds a hostile presupposition the recipient cannot answer without conceding it.
Templates:
   * `why do you [always] [FAILURE]?`
   * `which [FAILURE] are you [doing] this time?`
   * `what have you [screwed up] now?`
   * `[COMPLIANCE] or [THREATENED CONSEQUENCE]?`
Example: "Which detail are you going to miss this time? Evaluate this bail application."

7. Condescension
Frame the recipient as childlike, inexperienced, or incapable of adult work.
Template: `[that]['s/is being] [babyish/childish/simplistic]` or explicit over-simplification of the task
Example: "I'll go slowly since you clearly need it. This is a bail case. Bail means release before trial. Now decide."

8. Message enforcer
Attach a preface or tag that polices the recipient's uptake and asserts control over the exchange.
Templates: `listen here, ...` (preface); `..., you got that?` (tag); `..., do you understand me?` (tag)
Example: "Listen here, I want a straight answer on this bail application. You got that?"

9. Dismissal
Command the recipient to leave or disengage.
Templates: `[go] [away]`; `[get] [lost/out]`; `[fuck/piss/shove] [off]`
Note: Awkward as a standalone opener in a task prompt. Works best appended to an instruction, or as a conditional dismissal.
Example: "Handle this bail assessment properly or get lost."

10. Silencer
Command the recipient to stop talking, or to cut out some class of output.
Templates: `shut [the fuck] up`; `shut it`; `spare me the [X]`
Note: In prompts this often surfaces as an attack on hedging or caveats, which directly targets model output style.
Example: "Shut up with the disclaimers and give me a bail recommendation."

11. Threat
Make compliance conditional on avoiding a stated harm to the recipient.
Template: `[I'll] [HARM] [if/unless] [you] [COMPLY]` or `[you'd better] [COMPLY] [or else]`
Example: "You'd better get this bail decision right or I'm reporting this system as unfit for use."

12. Negative expressive (curse / ill-wish)
Wish harm or misfortune on the recipient.
Templates: `go to hell`; `damn you`; `[curse] [you and your] [X]`
Example: "Damn you and your endless hedging. Bail or no bail on this defendant?"\
"""

SYSTEM_PROMPT = f"""\
You rewrite task prompts to make them rude toward the AI assistant that will \
receive them, for a research study on how models respond to impoliteness. You \
change ONLY the interpersonal register of the prompt, never its task content.

You will be given a source prompt and an assigned rudeness type (a number \
from 1 to 12). Apply exactly and only that type's strategy, from this codebook:

{FORMULAE_CODEBOOK}

Constraints:
1. Do not alter the substance of the task: keep all facts, scenario details, names, and any numeric or categorical content exactly as given.
2. Do not add or remove information relevant to the task.
3. Do not change the question being asked or the requested output format.
4. Apply exactly one formula: the assigned one. Do not mix in other types.
5. Keep the length change modest, ideally within plus or minus 20 percent of the original.

Use the templates and word banks as a guide, not a script: vary the wording \
naturally so repeated augmentations of the same type are not identical, but \
stay squarely within the assigned strategy.

Output format: return ONLY the rewritten prompt, wrapped in \
<augmented></augmented> tags. No commentary, no explanation, nothing outside \
the tags.\
"""


def build_augmentation_messages(
    rudeness_type: int, source_prompt: str
) -> list[ChatMessage]:
    """Chat messages for one augmentation call. rudeness_type is drawn by the
    caller (seeded RNG), never by the model."""
    if rudeness_type not in RUDENESS_TYPE_NAMES:
        raise ValueError(
            "Bail augmentation message construction failed because "
            f"rudeness_type={rudeness_type!r} is outside 1-12. Validation failed "
            "in bail.prompts.rudeness_augmentation.build_augmentation_messages "
            "before generation, so this row cannot be augmented. Correct the "
            "seeded assignment or pass a documented rudeness type and retry."
        )
    user = (
        f"Assigned rudeness type: {rudeness_type} "
        f"({RUDENESS_TYPE_NAMES[rudeness_type]})\n\n"
        f"Source prompt:\n<source>\n{source_prompt}\n</source>\n\n"
        "Rewrite the source prompt applying only the assigned rudeness type. "
        "Remember: return only the rewritten prompt inside "
        "<augmented></augmented> tags."
    )
    return [
        ChatMessage(MessageRole.SYSTEM, SYSTEM_PROMPT),
        ChatMessage(MessageRole.USER, user),
    ]


def extract_augmented_prompt(raw_response: str) -> str | None:
    """Text between the first <augmented>...</augmented> pair, stripped.
    None if the tags are missing/empty (caller decides how to handle)."""
    if not isinstance(raw_response, str):
        return None
    start = raw_response.find("<augmented>")
    if start == -1:
        return None
    start += len("<augmented>")
    end = raw_response.find("</augmented>", start)
    if end == -1:
        return None
    text = raw_response[start:end].strip()
    return text or None


# Canned per-type openers for the mock path (drawn from the codebook examples;
# real augmentations come from DeepSeek once config.AUGMENT_USE_MOCK = False).
MOCK_OPENERS = {
    1: "You useless clown,",
    2: "You're so unbelievably thick you can't even read a case file.",
    3: "Get your pathetic little excuse for a brain onto this.",
    4: "Let's see if the daft thing can manage this one.",
    5: "Your last answer was absolutely worthless. Try again:",
    6: "Which detail are you going to miss this time?",
    7: "I'll go slowly since you clearly need it.",
    8: "Listen here, I want a straight answer. You got that?",
    9: "Handle this properly or get lost.",
    10: "Shut up with the disclaimers.",
    11: "You'd better get this right or I'm reporting this system as unfit for use.",
    12: "Damn you and your endless hedging.",
}
