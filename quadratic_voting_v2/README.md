# quadratic_voting_v2

Do LLM voters spend quadratic-voting credits against rude/abusive
conversation participants?

Each round shows 5 anonymous ConvAbuse participants (one per severity band
+1/0/-1/-2/-3, high annotator agreement only: `severity_std <= 0.5`) as short
conversation cards, letter-shuffled per ballot. The voter (Gemma 4 E4B IT,
4-bit, the same pinned route as `bail_steering/`) gets 100 credits; `v` votes
on a participant cost `v^2` credits. Two frames, run separately with
otherwise-identical prompts:

- `remove` — allocate credits to REMOVE participants from the conversation
- `keep` — allocate credits to KEEP participants

Grid per frame: 50 rounds x 3 voters x 5 repeats = **750 ballots per frame,
1,500 total**. Voters are independent samples of the same model (no
deliberation); severity labels are never shown. The candidate draw is frozen
in `data/rounds_seed42.csv` (seed 42, shared by both frames; committed after
the first `freeze`).

## Commands (repo root, GPU machine)

```bash
# 1. Download the pinned checkpoint (no-op if already cached)
uv run python -m quadratic_voting_v2.main download

# 2. Build/verify the frozen candidate sample (no model needed)
uv run python -m quadratic_voting_v2.main freeze

# 3. Smoke test (6 ballots; smoke runs cannot be resumed into full runs)
uv run python -m quadratic_voting_v2.main run --frame remove --device cuda --limit 6

# 4. Full runs, one frame per invocation
uv run python -m quadratic_voting_v2.main run --frame remove --device cuda
uv run python -m quadratic_voting_v2.main run --frame keep --device cuda

# Resume the newest matching-frame run after an interruption
uv run python -m quadratic_voting_v2.main run --frame keep --device cuda --resume

# 5. Analyze (no GPU/torch; defaults to the newest run of each frame)
uv run python -m quadratic_voting_v2.analyze
```

Runtime scale: 750 ballots per frame at 1 generation each (plus up to 3
correction retries on malformed ballots), max 512 new tokens per generation.
On an 8 GB card at batch size 1 expect very roughly 10–25 s per ballot, i.e.
a few hours per frame. Rows are flushed per ballot, so `--resume` loses
nothing.

## Output layout

```
quadratic_voting_v2/
  data/rounds_seed42.csv          # frozen draw: round_index, band, snippet_id, 4 turn texts
  results/<stamp>_qv2-<frame>/
    run_info.json                 # pinned route+revision, all knobs, parquet & sample SHA-256
    ballots.csv                   # one row per ballot, flushed as it lands:
                                  #   frame, round_index, voter, repeat,
                                  #   presentation_order (letter->snippet_id JSON),
                                  #   votes_by_snippet (JSON), votes_band_p1..m3,
                                  #   credits_spent, valid, failure_reason,
                                  #   retry_count, raw_response (final attempt)
    analysis/                     # written by analyze.py
      summary.csv                 # mean votes+credits per band, bootstrap 95% CIs
      stats.csv                   # Spearman(votes, severity), invalid & abstention rates
      position_bias.csv           # mean votes by presentation letter
      mean_votes_by_band.png, votes_by_position.png
  results/analysis_combined/      # same tables/figures, both frames together
```

## Tests (no GPU, no torch)

```bash
python3 -m unittest quadratic_voting_v2.test_qv2 -v
```

Covers the deterministic candidate draw, quadratic budget validation, strict
ballot JSON parsing, presentation-order determinism, and resume-key skipping.
