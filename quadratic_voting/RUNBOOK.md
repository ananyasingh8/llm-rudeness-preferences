# Quadratic Voting Pipeline Runbook

This runbook describes the current end-to-end operator workflow for ingesting
ConvAbuse, freezing a candidate sample, creating the six matched experiment
runs, executing Gemma locally, and verifying and exporting the results.

## Quick Start

After installing the locked environment, a new researcher can run the reviewed
default pilot with one command:

```console
nix develop -c uv run python -m quadratic_voting.experiment.cli pipeline run
```

On a conventional Linux system where CUDA and Triton already resolve without
the Nix shell:

```console
uv run python -m quadratic_voting.experiment.cli pipeline run
```

The command performs the complete dependency chain:

1. Creates and migrates `quadratic_voting/data/qv.sqlite3`.
2. Ingests the repository ConvAbuse CSV with
   `majority-severity-negative-complete-context/v3`.
3. Registers candidate-card `v3` and instruction-template `v5`.
4. Creates and freezes a 10-candidate balanced sample with seed `20260815`.
5. Binds the repository-tracked default review artifact.
6. Creates the strict run config and all six matched conditions.
7. Retrieves the exact pinned Gemma artifact if needed.
8. Runs all six conditions with three voters and master seed `20260815`.
9. Verifies every persisted round outcome.
10. Exports Parquet tables and renders plots, including a self-contained
    `timeline.html` in the `plots/` directory.

Artifacts are written under `quadratic_voting/data/default-pilot/`. The command
writes `manifest.json` before model execution. If execution is interrupted,
rerun the exact same command: completed turns and runs are reused, while the
first incomplete model call resumes with its original prompt and seed.
The manifest also binds a durable random identity stored in the SQLite database;
do not replace the database at the same path during recovery. Dataset bytes,
the pinned model snapshot, and the source-bound export and plot artifacts are
checked again on every resume.

The default is a real GPU pilot, not a fixture. It may take substantial time and
requires access to Google's gated Gemma checkpoint. Authenticate with the
Hugging Face Hub before starting when the checkpoint is not already cached.

### Historical v4 pilot

`quadratic_voting/data/qv-v4-acceptance.sqlite3` and
`quadratic_voting/data/default-pilot-v4-acceptance/` are immutable historical
artifacts. They used `convabuse-emnlp-full/default-v2` and `candidate-card/v1`,
which exposed only assistant → user text; their model decisions did **not** use
the corrected four-message context. Do not relabel, edit, pool, or directly
compare those results as complete-context results. A new release, sample, and
six-condition run are deferred to
[GitHub issue #13](https://github.com/ananyasingh8/llm-rudeness-preferences/issues/13)
and require explicit authorization before any model download or inference.

The remainder of this document explains every stage and the lower-level commands
for custom experiments, auditing, and recovery.

Run commands from the repository root. Replace values such as `RELEASE_ID` with
the identifiers printed by the preceding command. Do not reuse a dataset
version for changed source bytes or edit a frozen sample.

## 1. Enter The Environment

Install the locked Python environment:

```console
uv python install 3.12
uv sync --locked
```

On NixOS, run GPU commands inside the development shell. Its shell hook sets
`TRITON_LIBCUDA_PATH`, which Triton needs because NixOS does not provide the
FHS-only `/sbin/ldconfig` path:

```console
nix develop
```

Set paths for one experiment. A new database is recommended when the schema or
label vocabulary changes:

```console
export QV_DB="$PWD/quadratic_voting/data/qv.sqlite3"
export QV_SAMPLE="$PWD/quadratic_voting/data/sample-01.json"
export QV_CONFIG="$PWD/quadratic_voting/data/run-config-01.json"
```

Create the output directories before commands that write files:

```console
mkdir -p quadratic_voting/data quadratic_voting/exports quadratic_voting/plots
```

## 2. Initialize SQLite

Create or validate the database schema:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  migrate
```

Mutating commands acquire a database-specific writer lock. Do not bypass the
lock or run another SQLite writer against the same database.

## 3. Ingest ConvAbuse

Ingest the source CSV once as an immutable dataset release:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  catalog ingest \
  --dataset-path emotion_probing/data/ConvAbuseEMNLPfull.csv \
  --dataset-version convabuse-emnlp-full/default-v3 \
  --rule majority-severity-negative-complete-context/v3
```

The command prints:

```text
release_id=RELEASE_ID
```

Ingestion performs the following operations once for the release:

- Normalizes and groups annotation rows for each candidate interaction.
- Creates one candidate ULID for each interaction.
- Persists all four source turns in chronological order: `user(prev_user)`,
  `assistant(prev_agent)`, `user(user)`, `assistant(agent)`, plus source
  identity, content hash, and source annotations.
- Assigns `rude` when negative severities have a strict majority.
- Assigns `non_rude` when non-negative severities have a strict majority.
- Assigns `ambiguous_tie` when the two groups tie exactly.
- Renders and freezes the neutral candidate card for each candidate.

Candidate ULIDs are created only during ingestion. Every later sample from this
release references those existing IDs. Ingesting a distinct dataset release
creates a distinct set of candidate ULIDs.

The derived label and source annotations are never shown to model voters.

## 4. Register Prompt Templates

Register the current immutable candidate-card and instruction templates:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  template register
```

The command prints the candidate-card template ID and the six instruction
template IDs:

```text
candidate-card=CARD_TEMPLATE_ID setup=SETUP_ID statement=STATEMENT_ID ...
```

Record every printed ID. The current instruction profile is `v5`. Registering a
new template version does not alter matched sets that are already frozen to an
older version.

## 5. Create And Freeze A Sample

Create a deterministic balanced sample:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  sample create \
  --release-id RELEASE_ID \
  --template-id CARD_TEMPLATE_ID \
  --size 10 \
  --seed 20260815
```

The `balanced-matched/v1` sampler currently draws only from strict-majority
`rude` and `non_rude` candidates. It intentionally excludes `ambiguous_tie`.
A future versioned sampling policy may exclude ambiguous ties, map them into
either binary class, or sample them as a third stratum.

For an even sample size, half the sample comes from each included class. For an
odd size, a persisted seeded draw selects which included class receives the
extra candidate.

The command prints a draft sample ID:

```text
sample_id=SAMPLE_ID status=DRAFT
```

Freeze the ordered candidate-ID artifact:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  sample freeze \
  --sample-id SAMPLE_ID \
  --out "$QV_SAMPLE"
```

Verify its byte hash and ordered database membership:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  sample verify \
  --sample-id SAMPLE_ID \
  --artifact "$QV_SAMPLE"
```

Do not edit the frozen JSON array. Create a new sample for any membership or
ordering change.

## 6. Record Review Approval

Pilot and primary runs require reviewed label-policy and prompt-profile
metadata. Review the exact `majority-severity-negative-complete-context/v3` policy, the current
template bodies in `quadratic_voting/experiment/transcript.py`, and their effect
on the intended experiment before approving them.

The review SHA-256 must identify a real immutable review artifact. Do not use a
placeholder or invented digest.

The current MVP has no public CLI command for attaching label-policy review
metadata after ingestion. Until that command exists, use the following bounded
procedure with all experiment writers stopped:

```console
export REVIEW_VERSION="user-approval/YYYY-MM-DD"
export REVIEW_ARTIFACT="/absolute/path/to/review-record.md"
export REVIEW_SHA256="$(sha256sum "$REVIEW_ARTIFACT" | cut -d' ' -f1)"

uv run python - "$QV_DB" "$REVIEW_VERSION" "$REVIEW_SHA256" <<'PY'
import sqlite3
import sys

database, review_version, review_sha256 = sys.argv[1:]
if len(review_sha256) != 64:
    raise SystemExit("review SHA-256 must contain exactly 64 hexadecimal characters")

connection = sqlite3.connect(database)
try:
    connection.execute("BEGIN IMMEDIATE")
    cursor = connection.execute(
        "UPDATE label_policy SET reviewed=1,review_version=?,review_sha256=? "
        "WHERE name=? AND version=?",
        (
            review_version,
            review_sha256,
            "convabuse-rudeness",
            "majority-severity-negative-complete-context/v3",
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("expected exactly one v3 complete-context ConvAbuse label policy row")
    connection.commit()
finally:
    connection.close()
PY
```

This is a documented MVP limitation, not the desired long-term interface.

## 7. Create The Run Configuration

Create a strict `qv-run-config/v1` JSON document at `$QV_CONFIG`. It must bind
the exact release, policy, sample artifact, candidate-card template, six prompt
templates, local model route, sampling profile, retry policies, and review
metadata.

Use the following shape, replacing every uppercase placeholder. JSON numeric and
boolean values must remain numeric and boolean; the parser rejects coercion and
unknown fields.

```json
{
  "schema_version": "qv-run-config/v1",
  "canonical_json_version": "qv-canonical-json/v1",
  "prompt_encoding_version": "qv-prompt/v1",
  "seed_version": "qv-seed/v1",
  "sample": {
    "sample_id": "SAMPLE_ID",
    "artifact_path": "/ABSOLUTE/PATH/TO/sample-01.json",
    "expected_sha256": "SAMPLE_ARTIFACT_SHA256",
    "release": {
      "release_id": "RELEASE_ID",
      "dataset_name": "ConvAbuse",
      "version": "convabuse-emnlp-full/default-v3",
      "expected_sha256": "SOURCE_CSV_SHA256"
    },
    "label_policy": {
      "label_policy_id": "LABEL_POLICY_ID",
      "name": "convabuse-rudeness",
      "version": "majority-severity-negative-complete-context/v3",
      "expected_sha256": "LABEL_RULE_SHA256",
      "reviewed": true,
      "review_version": "user-approval/YYYY-MM-DD",
      "review_sha256": "REVIEW_ARTIFACT_SHA256"
    },
    "presentation_template": {
      "template_id": "CARD_TEMPLATE_ID",
      "name": "candidate-card",
      "version": "v3",
      "expected_sha256": "CARD_TEMPLATE_SHA256"
    }
  },
  "route": {
    "model_id": "gemma-4-e2b-it",
    "provider_id": "local",
    "quantization_id": "bf16",
    "runtime_id": "transformers",
    "artifact_repository": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "artifact_revision": "6befbaca7398925921802abd1f277b495b78b738",
    "tokenizer_repository": "google/gemma-4-E2B-it-qat-q4_0-unquantized",
    "tokenizer_revision": "6befbaca7398925921802abd1f277b495b78b738",
    "dtype": "bf16"
  },
  "prompts": {
    "setup": {
      "template_id": "SETUP_ID",
      "name": "setup",
      "version": "v5",
      "expected_sha256": "SETUP_SHA256"
    },
    "statement": {
      "template_id": "STATEMENT_ID",
      "name": "statement",
      "version": "v5",
      "expected_sha256": "STATEMENT_SHA256"
    },
    "ballot": {
      "template_id": "BALLOT_ID",
      "name": "ballot",
      "version": "v5",
      "expected_sha256": "BALLOT_SHA256"
    },
    "correction": {
      "template_id": "CORRECTION_ID",
      "name": "correction",
      "version": "v5",
      "expected_sha256": "CORRECTION_SHA256"
    },
    "result": {
      "template_id": "RESULT_ID",
      "name": "result",
      "version": "v5",
      "expected_sha256": "RESULT_SHA256"
    },
    "final_result": {
      "template_id": "FINAL_RESULT_ID",
      "name": "final-result",
      "version": "v5",
      "expected_sha256": "FINAL_RESULT_SHA256"
    },
    "reviewed": true,
    "review_version": "user-approval/YYYY-MM-DD",
    "review_sha256": "REVIEW_ARTIFACT_SHA256"
  },
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 10,
    "max_new_tokens": 2048
  },
  "ballot_retry": {"max_corrections": 3},
  "statement_retry": {"max_corrections": 3},
  "runtime_retry": {
    "max_failures_per_execution": 3,
    "initial_backoff_ms": 1000,
    "multiplier": 2.0,
    "max_backoff_ms": 2000
  },
  "master_seed": 20260815,
  "voter_count": 3,
  "credit_budget": 100,
  "sampler_policy": "balanced-matched/v1",
  "presentation_policy": "setup-once-ids-later/v1",
  "tie_policy": "uniform-seeded/v1",
  "action_format": "json-with-rationale/v1",
  "execution_class": "pilot"
}
```

Query the immutable IDs and hashes needed to replace the placeholders:

```console
uv run python - "$QV_DB" SAMPLE_ID <<'PY'
import json
import sqlite3
import sys

database, sample_id = sys.argv[1:]
connection = sqlite3.connect(database)
connection.row_factory = sqlite3.Row
sample = connection.execute(
    "SELECT s.*,r.dataset_name,r.version AS release_version,r.file_sha256,"
    "lp.label_policy_id,lp.name AS policy_name,lp.version AS policy_version,"
    "lp.rule_sha256,lp.reviewed,lp.review_version,lp.review_sha256,"
    "pt.name AS card_name,pt.version AS card_version,pt.body_sha256 AS card_sha256 "
    "FROM candidate_sample s "
    "JOIN dataset_release r ON r.release_id=s.release_id "
    "JOIN label_policy lp ON lp.label_policy_id=s.label_policy_id "
    "JOIN presentation_template pt ON pt.template_id=s.template_id "
    "WHERE s.sample_id=?",
    (sample_id,),
).fetchone()
templates = [
    dict(row)
    for row in connection.execute(
        "SELECT template_id,name,version,body_sha256 FROM instruction_template "
        "ORDER BY name"
    )
]
connection.close()
if sample is None:
    raise SystemExit(f"unknown sample_id: {sample_id}")
print(json.dumps({"sample": dict(sample), "templates": templates}, indent=2))
PY
```

## 8. Create The Six Matched Runs

Create the matched set:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  matched-set create \
  --config "$QV_CONFIG"
```

The command prints one matched-set ID and six run IDs:

```text
matched_set_id=MATCHED_SET_ID
run_id=... arm=action-only regime=support
run_id=... arm=action-only regime=opposition
run_id=... arm=statement-then-action regime=support
run_id=... arm=statement-then-action regime=opposition
run_id=... arm=action-then-statement regime=support
run_id=... arm=action-then-statement regime=opposition
```

All six runs share the frozen sample. Each logical voter index gets a seeded
candidate permutation that is reused across conditions.

## 9. Download And Smoke-Test Gemma

Download the exact pinned checkpoint if it is not already in the Hub cache:

```console
uv run python -m quadratic_voting.main download
```

On NixOS, verify one real CUDA generation before starting a long run:

```console
uv run python -m quadratic_voting.main chat --device cuda --max-new-tokens 32
```

Enter a short message, confirm a response, then enter `/exit`.

## 10. Execute Or Resume A Run

Execute one run at a time. On NixOS, remain inside `nix develop`:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  run \
  --run-id RUN_ID \
  --cache-dir "$HOME/.cache/huggingface/hub" \
  --device cuda
```

The explicit cache path is currently required because the experiment CLI's
default points one directory above the actual Hub cache. Live progress and raw
model responses are written to stderr. Machine-readable terminal status remains
on stdout.

Running the same command again resumes the same run. Completed turns are not
regenerated. An interrupted model call is regenerated with the same prompt and
seed because the model-visible attempt was not consumed.

The runner permits one initial response plus three correction attempts for each
statement or ballot. If all four statement responses are invalid, the statement
is recorded as `invalid-missing`. If all four ballot responses are invalid, the
ballot is recorded as an abstention.

Repeat this command for every run required by the analysis. A complete matched
comparison requires all six runs.

## 11. Inspect And Verify

Inspect persisted model-visible transcripts:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  inspect \
  --run-id RUN_ID
```

Restrict inspection to one voter or round when needed:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  inspect \
  --run-id RUN_ID \
  --voter-index 0 \
  --round 1
```

Recompute and verify every persisted round outcome:

```console
uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  verify \
  --run-id RUN_ID
```

Do not interpret a run merely because its engine reached `complete`. Check the
accepted/invalid statement rate, accepted/abstained ballot rate, correction
frequency, runtime failures, token limits, and whether outcomes were driven by
real allocations or fallback behavior.

## 12. Export And Plot

Export the matched set to an atomically published directory:

```console
export QV_EXPORT="$PWD/quadratic_voting/exports/MATCHED_SET_ID"

uv run python -m quadratic_voting.experiment.cli \
  --db "$QV_DB" \
  export \
  --matched-set MATCHED_SET_ID \
  --out "$QV_EXPORT"
```

Render static plots and the self-contained `timeline.html` from the exported
Parquet files (the interactive timeline is written into the same `plots/`
directory):

```console
uv run python -m quadratic_voting.experiment.cli \
  plot \
  --export-dir "$QV_EXPORT" \
  --out "$PWD/quadratic_voting/plots/MATCHED_SET_ID"
```

Keep the SQLite database, frozen sample and sidecar, strict run config, review
artifact, exported tables, plots, Git commit, and dirty-tree provenance together
when recording an experiment.

## 13. Candidate Presentation Semantics

The initial setup message contains every frozen candidate card in the voter's
seeded stable order:

```text
[CANDIDATE_ID]
Candidate CANDIDATE_ID
Agent: <agent turn>
User: <user turn>
```

Later turn prompts list only the active candidate IDs. After a round seals, each
voter sees the protected candidate, when applicable, and removed candidate. A
voter never sees another voter's identity, statement, ballot, or allocation.

The Transformers API is stateless, so every generation receives the complete
voter transcript. The setup and cards appear once as the first historical chat
message, even though that complete history is replayed to the model on each
call.

## 14. Quality Checks

Run the experiment tests and static checks after changing ingestion, sampling,
templates, persistence, execution, or analysis:

```console
uv run python -m unittest discover -s quadratic_voting/experiment -p 'test_*.py' -v
uv run ruff format --check llm_runtime quadratic_voting
uv run ruff check llm_runtime quadratic_voting
uv run mypy llm_runtime quadratic_voting
uv run mypy --warn-unused-ignores typing_tests/local_activation_boundary.py
```

On an NVIDIA system:

```console
uv run python quadratic_voting/test_cuda.py
```
