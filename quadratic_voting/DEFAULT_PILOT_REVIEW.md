# Default Pilot Review

Review version: `default-pilot-review/v1`

This artifact records the reviewed defaults used by `qv pipeline run`.

## Label Policy

Policy: `majority-severity-negative/v2`

- Severity annotations `-1`, `-2`, and `-3` count as negative.
- A strict negative majority is labeled `rude`.
- A strict non-negative majority is labeled `non_rude`.
- An exact tie is labeled `ambiguous_tie`.
- `balanced-matched/v1` excludes ambiguous ties. Future versioned sampling
  policies may explicitly map or sample them differently.
- Labels are used for sampling and analysis and are not shown to model voters.

## Prompt Profile

Profile: instruction templates `v3`, candidate card `v1`.

- Voters are told the regime, iterative process, hidden peer choices, active
  candidates, quadratic budget, and response order.
- Setup includes valid statement and ballot JSON examples.
- Responses must contain only the requested JSON object.
- Each turn permits one initial response and three correction attempts.
- Corrections restate the round, turn kind, active IDs, schema, exact failures,
  attempts remaining, and final invalid-response consequence.
- Exhausted statement retries produce `invalid-missing`; exhausted ballot
  retries produce an abstention.

## Default Pilot

- Six matched conditions: three elicitation arms by two voting regimes.
- 10 candidates: five strict-majority rude and five strict-majority non-rude.
- 3 logical voters.
- Candidate sample seed: `20260815`.
- Master experiment seed: `20260815`.
- 100 quadratic voice credits replenished each round.
- Gemma 4 E2B instruction-tuned BF16 route at the repository-pinned revision.
- Sampling: temperature `0.7`, top-p `0.9`, top-k `10`, and at most `8192`
  generated tokens.
