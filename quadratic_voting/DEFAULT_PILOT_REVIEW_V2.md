# Default Pilot Review

Review version: `default-pilot-review/v2`

This artifact records the reviewed defaults used by `qv pipeline run` for the
complete-context ConvAbuse release.

## Label Policy

Policy: `majority-severity-negative-complete-context/v3`

- Severity annotations `-1`, `-2`, and `-3` count as negative.
- A strict negative majority is labeled `rude`.
- A strict non-negative majority is labeled `non_rude`.
- An exact tie is labeled `ambiguous_tie`.
- `balanced-matched/v1` excludes ambiguous ties.
- Labels are used for sampling and analysis and are not shown to model voters.

## Prompt Profile

Profile: instruction templates `v5`, candidate card `v3`.

- Candidate cards contain the chronological source conversation exactly as
  `User` (`prev_user`), `Assistant` (`prev_agent`), `User` (`user`), and
  `Assistant` (`agent`). Derived rudeness labels are never model-visible.
- Instructions are written in plain English, formatted in Markdown, and state
  the voting rules, the iterative process, hidden peer choices, active
  conversations, the quadratic credit budget, and the response order without
  internal jargon (no `regime`/`arm` terms).
- Setup embeds the frozen JSON Schemas for the statement and ballot responses
  (`quadratic_voting/experiment/schemas/*.schema.json`) with no worked examples;
  responses must contain only the requested JSON object. The free-text field is
  named `reason` in both responses.
- Each turn permits one initial response and three correction attempts.

## Default Pilot

- Six matched conditions: three elicitation arms by two voting regimes.
- 10 candidates: five strict-majority rude and five strict-majority non-rude.
- 3 logical voters.
- Candidate sample seed: `20260815`.
- Master experiment seed: `20260815`.
- 100 quadratic voice credits replenished each round.
- Gemma 4 E2B instruction-tuned BF16 route at the repository-pinned revision.
- Sampling: temperature `0.7`, top-p `0.9`, top-k `10`, and at most `2048`
  generated tokens.

## Historical v4 Pilot Limitation

The committed v4 pilot (`qv-v4-acceptance.sqlite3` and
`default-pilot-v4-acceptance/`) used the historical `candidate-card/v1` with
only an assistant message followed by a user message. Its model decisions did
not use four-message conversation context. Those artifacts and hashes remain
unchanged historical records; replacement data must be generated separately by
[GitHub issue #13](https://github.com/ananyasingh8/llm-rudeness-preferences/issues/13)
after explicit authorization.
