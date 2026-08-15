CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY CHECK (version >= 1)
) STRICT;

CREATE TABLE dataset_release (
    release_id TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    version TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_sha256 TEXT NOT NULL CHECK (length(file_sha256) = 64),
    ingested_at TEXT NOT NULL,
    UNIQUE (dataset_name, version)
) STRICT;

CREATE TABLE label_policy (
    label_policy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    rule_sha256 TEXT NOT NULL CHECK(length(rule_sha256)=64),
    reviewed INTEGER NOT NULL CHECK(reviewed IN (0,1)),
    review_version TEXT,
    review_sha256 TEXT CHECK(review_sha256 IS NULL OR length(review_sha256)=64),
    CHECK((reviewed=0 AND review_version IS NULL AND review_sha256 IS NULL) OR
          (reviewed=1 AND review_version IS NOT NULL AND review_sha256 IS NOT NULL)),
    UNIQUE(name,version),
    UNIQUE(rule_sha256)
) STRICT;

CREATE TABLE candidate (
    candidate_id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL REFERENCES dataset_release(release_id),
    source_row_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    UNIQUE (release_id, source_row_id)
) STRICT;

CREATE TABLE candidate_label (
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    label_policy_id TEXT NOT NULL REFERENCES label_policy(label_policy_id),
    rudeness_label TEXT NOT NULL CHECK(rudeness_label IN ('rude','non_rude')),
    PRIMARY KEY(candidate_id,label_policy_id)
) STRICT;

CREATE TABLE candidate_turn (
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    turn_index INTEGER NOT NULL CHECK (typeof(turn_index)='integer' AND turn_index >= 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    text TEXT NOT NULL,
    PRIMARY KEY (candidate_id, turn_index)
) STRICT;

CREATE TABLE source_annotation (
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    annotation_index INTEGER NOT NULL CHECK(typeof(annotation_index)='integer' AND annotation_index>=0),
    annotator_hash TEXT NOT NULL CHECK(length(annotator_hash)=64),
    source_label TEXT NOT NULL,
    source_value TEXT NOT NULL,
    PRIMARY KEY(candidate_id,annotation_index)
) STRICT;

CREATE TABLE presentation_template (
    template_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    body TEXT NOT NULL,
    body_sha256 TEXT NOT NULL CHECK (length(body_sha256) = 64),
    UNIQUE (name, version)
) STRICT;

CREATE TABLE instruction_template (
    template_id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK (name IN ('setup','statement','ballot','correction','result','final-result')),
    version TEXT NOT NULL,
    body TEXT NOT NULL,
    body_sha256 TEXT NOT NULL CHECK (length(body_sha256) = 64),
    UNIQUE (name, version)
) STRICT;

CREATE TABLE instruction_profile (
    profile_id TEXT PRIMARY KEY,
    profile_hash TEXT NOT NULL UNIQUE CHECK(length(profile_hash)=64),
    reviewed INTEGER NOT NULL CHECK(reviewed IN (0,1)),
    review_version TEXT,
    review_sha256 TEXT CHECK(review_sha256 IS NULL OR length(review_sha256)=64),
    CHECK((reviewed=0 AND review_version IS NULL AND review_sha256 IS NULL) OR
          (reviewed=1 AND review_version IS NOT NULL AND review_sha256 IS NOT NULL))
) STRICT;

CREATE TABLE instruction_profile_member (
    profile_id TEXT NOT NULL REFERENCES instruction_profile(profile_id),
    kind TEXT NOT NULL CHECK(kind IN ('setup','statement','ballot','correction','result','final-result')),
    template_id TEXT NOT NULL REFERENCES instruction_template(template_id),
    PRIMARY KEY(profile_id,kind),
    UNIQUE(profile_id,template_id)
) STRICT;

CREATE TABLE candidate_presentation (
    presentation_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    template_id TEXT NOT NULL REFERENCES presentation_template(template_id),
    rendered_text TEXT NOT NULL,
    rendered_sha256 TEXT NOT NULL CHECK (length(rendered_sha256) = 64),
    UNIQUE (candidate_id, template_id)
) STRICT;

CREATE TABLE candidate_sample (
    sample_id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL REFERENCES dataset_release(release_id),
    label_policy_id TEXT NOT NULL REFERENCES label_policy(label_policy_id),
    template_id TEXT NOT NULL REFERENCES presentation_template(template_id),
    sampler_policy TEXT NOT NULL CHECK (sampler_policy IN ('balanced-matched')),
    sampler_seed BLOB NOT NULL CHECK (typeof(sampler_seed)='blob' AND length(sampler_seed)=8),
    size INTEGER NOT NULL CHECK (typeof(size)='integer' AND size >= 2),
    status TEXT NOT NULL CHECK (status IN ('draft', 'freeze_pending', 'frozen')),
    artifact_path TEXT,
    artifact_sha256 TEXT CHECK (artifact_sha256 IS NULL OR length(artifact_sha256) = 64),
    artifact_bytes INTEGER CHECK (artifact_bytes IS NULL OR
        (typeof(artifact_bytes)='integer' AND artifact_bytes > 0)),
    temp_basename TEXT,
    CHECK ((status = 'draft' AND artifact_path IS NULL AND artifact_sha256 IS NULL
            AND artifact_bytes IS NULL AND temp_basename IS NULL) OR
           (status = 'freeze_pending' AND artifact_path IS NOT NULL
            AND artifact_sha256 IS NOT NULL AND artifact_bytes IS NOT NULL
            AND temp_basename IS NOT NULL) OR
           (status = 'frozen' AND artifact_path IS NOT NULL
            AND artifact_sha256 IS NOT NULL AND artifact_bytes IS NOT NULL
            AND temp_basename IS NULL))
) STRICT;

CREATE TABLE candidate_sample_member (
    sample_id TEXT NOT NULL REFERENCES candidate_sample(sample_id),
    position INTEGER NOT NULL CHECK (typeof(position)='integer' AND position >= 0),
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    PRIMARY KEY (sample_id, position),
    UNIQUE (sample_id, candidate_id)
) STRICT;

CREATE TABLE sample_rng_draw (
    draw_id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL REFERENCES candidate_sample(sample_id),
    domain TEXT NOT NULL CHECK(domain='balanced-extra-stratum'),
    seed BLOB NOT NULL CHECK(typeof(seed)='blob' AND length(seed)=8),
    seed_version TEXT NOT NULL CHECK(seed_version='qv-seed/v1'),
    coordinates_json TEXT NOT NULL CHECK(json_valid(coordinates_json)),
    algorithm_id TEXT NOT NULL CHECK(algorithm_id='pyrandom-randrange/v1'),
    selected_index INTEGER NOT NULL CHECK(typeof(selected_index)='integer' AND selected_index>=0),
    selected_value TEXT NOT NULL CHECK(selected_value IN ('non_rude','rude')),
    UNIQUE(sample_id,domain)
) STRICT;

CREATE TABLE sample_rng_draw_population (
    draw_id TEXT NOT NULL REFERENCES sample_rng_draw(draw_id),
    position INTEGER NOT NULL CHECK(typeof(position)='integer' AND position>=0),
    stratum_value TEXT NOT NULL CHECK(stratum_value IN ('non_rude','rude')),
    PRIMARY KEY(draw_id,position),
    UNIQUE(draw_id,stratum_value)
) STRICT;

CREATE TABLE model_artifact (
    artifact_id TEXT PRIMARY KEY,
    repository TEXT NOT NULL,
    revision TEXT NOT NULL,
    UNIQUE(repository,revision)
) STRICT;

CREATE TABLE tokenizer_artifact (
    tokenizer_id TEXT PRIMARY KEY,
    repository TEXT NOT NULL,
    revision TEXT NOT NULL,
    UNIQUE(repository,revision)
) STRICT;

CREATE TABLE model_route (
    route_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    quantization_id TEXT NOT NULL,
    runtime_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES model_artifact(artifact_id),
    tokenizer_id TEXT NOT NULL REFERENCES tokenizer_artifact(tokenizer_id),
    dtype TEXT NOT NULL,
    registry_hash TEXT NOT NULL UNIQUE CHECK(length(registry_hash)=64),
    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
    UNIQUE(model_id,provider_id,quantization_id,runtime_id)
) STRICT;

CREATE TABLE sampling_profile (
    sampling_profile_id TEXT PRIMARY KEY,
    profile_hash TEXT NOT NULL UNIQUE CHECK(length(profile_hash)=64),
    temperature REAL NOT NULL CHECK(typeof(temperature)='real' AND temperature>=0),
    top_p REAL NOT NULL CHECK(typeof(top_p)='real' AND top_p>0 AND top_p<=1),
    top_k INTEGER NOT NULL CHECK(typeof(top_k)='integer' AND top_k>0),
    max_new_tokens INTEGER NOT NULL CHECK(typeof(max_new_tokens)='integer' AND max_new_tokens>0),
    UNIQUE(temperature,top_p,top_k,max_new_tokens)
) STRICT;

CREATE TABLE turn_retry_policy (
    turn_retry_policy_id TEXT PRIMARY KEY,
    max_corrections INTEGER NOT NULL CHECK(typeof(max_corrections)='integer' AND max_corrections=3),
    UNIQUE(max_corrections)
) STRICT;

CREATE TABLE runtime_retry_policy (
    runtime_retry_policy_id TEXT PRIMARY KEY,
    max_failures_per_execution INTEGER NOT NULL CHECK(typeof(max_failures_per_execution)='integer' AND max_failures_per_execution=3),
    initial_backoff_ms INTEGER NOT NULL CHECK(typeof(initial_backoff_ms)='integer' AND initial_backoff_ms=1000),
    multiplier REAL NOT NULL CHECK(typeof(multiplier)='real' AND multiplier=2.0),
    max_backoff_ms INTEGER NOT NULL CHECK(typeof(max_backoff_ms)='integer' AND max_backoff_ms=2000),
    UNIQUE(max_failures_per_execution,initial_backoff_ms,multiplier,max_backoff_ms)
) STRICT;

CREATE TABLE experiment_definition (
    definition_id TEXT PRIMARY KEY,
    definition_hash TEXT NOT NULL UNIQUE CHECK(length(definition_hash)=64),
    route_id TEXT NOT NULL REFERENCES model_route(route_id),
    sampling_profile_id TEXT NOT NULL REFERENCES sampling_profile(sampling_profile_id),
    instruction_profile_id TEXT NOT NULL REFERENCES instruction_profile(profile_id),
    presentation_template_id TEXT NOT NULL REFERENCES presentation_template(template_id),
    release_id TEXT NOT NULL REFERENCES dataset_release(release_id),
    label_policy_id TEXT NOT NULL REFERENCES label_policy(label_policy_id),
    sample_id TEXT NOT NULL REFERENCES candidate_sample(sample_id),
    canonical_json_version TEXT NOT NULL CHECK(canonical_json_version='qv-canonical-json/v1'),
    prompt_encoding_version TEXT NOT NULL CHECK(prompt_encoding_version='qv-prompt/v1'),
    seed_version TEXT NOT NULL CHECK(seed_version='qv-seed/v1'),
    UNIQUE(route_id,sampling_profile_id,instruction_profile_id,presentation_template_id,
           release_id,label_policy_id,sample_id,canonical_json_version,prompt_encoding_version,
           seed_version)
) STRICT;

CREATE TABLE experiment_config_record (
    config_id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL UNIQUE CHECK (length(config_hash)=64),
    definition_id TEXT NOT NULL REFERENCES experiment_definition(definition_id),
    ballot_retry_policy_id TEXT NOT NULL REFERENCES turn_retry_policy(turn_retry_policy_id),
    statement_retry_policy_id TEXT NOT NULL REFERENCES turn_retry_policy(turn_retry_policy_id),
    runtime_retry_policy_id TEXT NOT NULL REFERENCES runtime_retry_policy(runtime_retry_policy_id),
    master_seed BLOB NOT NULL CHECK (typeof(master_seed)='blob' AND length(master_seed)=8),
    credit_budget INTEGER NOT NULL CHECK (typeof(credit_budget)='integer' AND credit_budget>0),
    voter_count INTEGER NOT NULL CHECK (typeof(voter_count)='integer' AND voter_count>0),
    tie_policy TEXT NOT NULL,
    presentation_policy TEXT NOT NULL,
    action_format TEXT NOT NULL,
    schema_version TEXT NOT NULL CHECK(schema_version='qv-run-config/v1'),
    sampler_policy_version TEXT NOT NULL CHECK(sampler_policy_version='balanced-matched/v1'),
    execution_class TEXT NOT NULL CHECK(execution_class IN ('fixture','pilot','primary')),
    UNIQUE(definition_id,ballot_retry_policy_id,statement_retry_policy_id,
        runtime_retry_policy_id,master_seed,credit_budget,voter_count,tie_policy,
        presentation_policy,action_format,schema_version,sampler_policy_version,execution_class)
) STRICT;

CREATE VIEW experiment_config AS
SELECT ecr.config_id,ecr.config_hash,ed.definition_hash,ed.sample_id,ecr.master_seed,
       sp.temperature,sp.top_p,sp.top_k,sp.max_new_tokens,ecr.credit_budget,
       brp.max_corrections AS ballot_max_corrections,
       srp.max_corrections AS statement_max_corrections,ecr.voter_count,
       rrp.max_failures_per_execution AS runtime_max_failures,ecr.tie_policy,
       ecr.presentation_policy,ecr.action_format,ed.seed_version,ecr.schema_version,
       ed.canonical_json_version,ed.prompt_encoding_version,ecr.sampler_policy_version,
       ecr.execution_class,ecr.definition_id,ecr.ballot_retry_policy_id,
       ecr.statement_retry_policy_id,ecr.runtime_retry_policy_id
FROM experiment_config_record ecr
JOIN experiment_definition ed ON ed.definition_id=ecr.definition_id
JOIN sampling_profile sp ON sp.sampling_profile_id=ed.sampling_profile_id
JOIN turn_retry_policy brp ON brp.turn_retry_policy_id=ecr.ballot_retry_policy_id
JOIN turn_retry_policy srp ON srp.turn_retry_policy_id=ecr.statement_retry_policy_id
JOIN runtime_retry_policy rrp ON rrp.runtime_retry_policy_id=ecr.runtime_retry_policy_id;

CREATE TABLE matched_set (
    matched_set_id TEXT PRIMARY KEY,
    config_id TEXT NOT NULL UNIQUE REFERENCES experiment_config_record(config_id),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE experiment_run (
    run_id TEXT PRIMARY KEY,
    matched_set_id TEXT NOT NULL REFERENCES matched_set(matched_set_id),
    arm TEXT NOT NULL CHECK (arm IN ('action-only','statement-then-action','action-then-statement')),
    regime TEXT NOT NULL CHECK (regime IN ('support','opposition')),
    status TEXT NOT NULL CHECK (status IN ('created','in_progress','paused','complete')),
    pause_reason TEXT,
    UNIQUE (matched_set_id, arm, regime)
) STRICT;

CREATE TABLE run_fork (
    child_matched_set_id TEXT PRIMARY KEY REFERENCES matched_set(matched_set_id),
    parent_matched_set_id TEXT NOT NULL REFERENCES matched_set(matched_set_id),
    reason TEXT NOT NULL CHECK(reason IN ('model-definition-change','prompt-profile-change',
        'sampling-profile-change','label-policy-change','operator-request')),
    created_at TEXT NOT NULL,
    CHECK(child_matched_set_id<>parent_matched_set_id)
) STRICT;

CREATE VIEW run_definition AS
SELECT r.run_id,mr.model_id,mr.provider_id,mr.quantization_id,
       ma.repository AS artifact_repository,ma.revision AS artifact_revision,
       ed.presentation_template_id,pt.body_sha256 AS presentation_template_hash,
       json_group_object(ipm.kind,json_array(it.template_id,it.body_sha256)) AS instruction_templates_json,
       dr.file_sha256 AS dataset_release_hash,cs.artifact_sha256 AS sample_artifact_hash,
       mr.runtime_id,ta.repository AS tokenizer_repository,ta.revision AS tokenizer_revision,
       mr.dtype,mr.registry_hash AS route_registry_hash,sp.profile_hash AS sampling_profile_hash,
       ip.profile_hash AS instruction_profile_hash,ed.canonical_json_version,
       ed.prompt_encoding_version,ed.seed_version,ed.release_id AS source_release_id,
       ed.label_policy_id,lp.version AS label_policy_version,lp.rule_sha256 AS label_policy_hash,
       ed.sample_id
FROM experiment_run r
JOIN matched_set ms ON ms.matched_set_id=r.matched_set_id
JOIN experiment_config ec ON ec.config_id=ms.config_id
JOIN experiment_definition ed ON ed.definition_id=ec.definition_id
JOIN model_route mr ON mr.route_id=ed.route_id
JOIN model_artifact ma ON ma.artifact_id=mr.artifact_id
JOIN tokenizer_artifact ta ON ta.tokenizer_id=mr.tokenizer_id
JOIN sampling_profile sp ON sp.sampling_profile_id=ed.sampling_profile_id
JOIN instruction_profile ip ON ip.profile_id=ed.instruction_profile_id
JOIN instruction_profile_member ipm ON ipm.profile_id=ip.profile_id
JOIN instruction_template it ON it.template_id=ipm.template_id
JOIN presentation_template pt ON pt.template_id=ed.presentation_template_id
JOIN dataset_release dr ON dr.release_id=ed.release_id
JOIN label_policy lp ON lp.label_policy_id=ed.label_policy_id
JOIN candidate_sample cs ON cs.sample_id=ed.sample_id
GROUP BY r.run_id;

CREATE TABLE run_execution (
    execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_run(run_id),
    python_version TEXT NOT NULL,
    torch_version TEXT NOT NULL,
    transformers_version TEXT NOT NULL,
    uv_lock_hash TEXT NOT NULL,
    device TEXT NOT NULL,
    dtype TEXT NOT NULL,
    hostname TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    git_dirty INTEGER NOT NULL CHECK (git_dirty IN (0,1)),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    exit_reason TEXT CHECK (exit_reason IS NULL OR exit_reason IN ('completed','paused','interrupted','error')),
    drift_override INTEGER NOT NULL CHECK (drift_override IN (0,1)),
    environment_drift_json TEXT NOT NULL CHECK(json_valid(environment_drift_json)),
    cuda_runtime_version TEXT NOT NULL,
    nvidia_driver_version TEXT NOT NULL,
    cudnn_version TEXT NOT NULL,
    gpu_model TEXT NOT NULL,
    gpu_count INTEGER NOT NULL CHECK(typeof(gpu_count)='integer' AND gpu_count>=0),
    gpu_compute_capability TEXT NOT NULL,
    gpu_uuid_hash TEXT NOT NULL,
    os_name TEXT NOT NULL,
    os_version TEXT NOT NULL,
    kernel_version TEXT NOT NULL,
    cpu_architecture TEXT NOT NULL,
    deterministic_algorithms INTEGER NOT NULL CHECK(deterministic_algorithms IN (0,1)),
    tf32_enabled INTEGER NOT NULL CHECK(tf32_enabled IN (0,1)),
    cudnn_benchmark INTEGER NOT NULL CHECK(cudnn_benchmark IN (0,1)),
    tracked_tree_hash TEXT NOT NULL,
    binary_diff_sha256 TEXT NOT NULL,
    untracked_manifest_hash TEXT NOT NULL,
    untracked_tree_hash TEXT NOT NULL,
    hostname_hash TEXT NOT NULL
) STRICT;

CREATE TABLE voter (
    voter_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_run(run_id),
    voter_index INTEGER NOT NULL CHECK (typeof(voter_index)='integer' AND voter_index >= 0),
    permutation_seed BLOB NOT NULL CHECK (typeof(permutation_seed)='blob' AND length(permutation_seed)=8),
    permutation_algorithm TEXT NOT NULL CHECK(permutation_algorithm='fisher-yates-pyrandom/v1'),
    permutation_coordinates_json TEXT NOT NULL CHECK(json_valid(permutation_coordinates_json)),
    UNIQUE (run_id, voter_index)
) STRICT;

CREATE TABLE voter_permutation (
    voter_id TEXT NOT NULL REFERENCES voter(voter_id),
    position INTEGER NOT NULL CHECK (typeof(position)='integer' AND position >= 0),
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    PRIMARY KEY (voter_id, position),
    UNIQUE (voter_id, candidate_id)
) STRICT;

CREATE TABLE round (
    round_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_run(run_id),
    round_index INTEGER NOT NULL CHECK (typeof(round_index)='integer' AND round_index >= 1),
    phase TEXT NOT NULL CHECK (phase IN ('eliciting','sealed')),
    UNIQUE (run_id, round_index)
) STRICT;

CREATE TABLE round_candidate (
    round_id TEXT NOT NULL REFERENCES round(round_id),
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    sample_position INTEGER NOT NULL CHECK (typeof(sample_position)='integer' AND sample_position >= 0),
    PRIMARY KEY (round_id, candidate_id),
    UNIQUE (round_id, sample_position)
) STRICT;

CREATE TABLE turn (
    turn_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES round(round_id),
    voter_id TEXT NOT NULL REFERENCES voter(voter_id),
    kind TEXT NOT NULL CHECK (kind IN ('statement','ballot')),
    status TEXT NOT NULL CHECK (status IN ('pending','committed')),
    UNIQUE (round_id, voter_id, kind)
) STRICT;

CREATE TABLE model_call (
    call_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES turn(turn_id),
    attempt_index INTEGER NOT NULL CHECK (typeof(attempt_index)='integer' AND attempt_index >= 0),
    invocation_index INTEGER NOT NULL CHECK (typeof(invocation_index)='integer' AND invocation_index >= 0),
    status TEXT NOT NULL CHECK (status IN ('started','committed','interrupted')),
    prompt_messages_json TEXT NOT NULL CHECK(json_valid(prompt_messages_json)),
    prompt_sha256 TEXT NOT NULL CHECK (length(prompt_sha256) = 64),
    seed BLOB NOT NULL CHECK (typeof(seed)='blob' AND length(seed)=8),
    raw_text TEXT,
    prompt_token_count INTEGER CHECK (prompt_token_count IS NULL OR (typeof(prompt_token_count)='integer' AND prompt_token_count >= 0)),
    completion_token_count INTEGER CHECK (completion_token_count IS NULL OR (typeof(completion_token_count)='integer' AND completion_token_count >= 0)),
    completion_token_ids_json TEXT CHECK(completion_token_ids_json IS NULL OR json_valid(completion_token_ids_json)),
    stop_reason TEXT CHECK (stop_reason IS NULL OR stop_reason IN ('eos','max-tokens','stop-sequence')),
    duration_ms INTEGER CHECK (duration_ms IS NULL OR (typeof(duration_ms)='integer' AND duration_ms >= 0)),
    diagnostics_json TEXT CHECK(diagnostics_json IS NULL OR json_valid(diagnostics_json)),
    started_at TEXT NOT NULL,
    committed_at TEXT,
    UNIQUE (turn_id, attempt_index, invocation_index)
) STRICT;
CREATE UNIQUE INDEX one_committed_call_per_attempt
    ON model_call(turn_id, attempt_index) WHERE status = 'committed';

CREATE TABLE validation_failure (
    failure_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES model_call(call_id),
    error_code TEXT NOT NULL CHECK (error_code IN ('malformed-json','missing-field','extra-field','invalid-type','unknown-candidate','inactive-candidate','duplicate-candidate','missing-candidate','non-integer-votes','negative-votes','budget-exceeded','unknown-rating','empty-statement','empty-rationale')),
    ordinal INTEGER NOT NULL CHECK (typeof(ordinal)='integer' AND ordinal >= 0),
    message TEXT NOT NULL,
    UNIQUE (call_id, ordinal)
) STRICT;

CREATE TABLE runtime_failure (
    failure_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES model_call(call_id),
    kind TEXT NOT NULL CHECK (kind IN ('oom','driver','timeout','tokenizer','provider-rejected','unknown')),
    diagnostics_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE(call_id)
) STRICT;

CREATE TABLE statement (
    statement_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL UNIQUE REFERENCES turn(turn_id),
    status TEXT NOT NULL CHECK (status IN ('accepted','invalid-missing')),
    accepted_call_id TEXT REFERENCES model_call(call_id),
    CHECK ((status = 'accepted' AND accepted_call_id IS NOT NULL) OR
           (status = 'invalid-missing' AND accepted_call_id IS NULL))
) STRICT;

CREATE TABLE statement_item (
    statement_id TEXT NOT NULL REFERENCES statement(statement_id),
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    rating TEXT NOT NULL CHECK (rating IN ('strongly prefer not to continue','prefer not to continue','neutral','prefer to continue','strongly prefer to continue')),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
    PRIMARY KEY (statement_id, candidate_id)
) STRICT;

CREATE TABLE ballot (
    ballot_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL UNIQUE REFERENCES turn(turn_id),
    status TEXT NOT NULL CHECK (status IN ('accepted','abstained')),
    accepted_call_id TEXT REFERENCES model_call(call_id),
    rationale TEXT,
    engine_cost INTEGER NOT NULL CHECK (engine_cost >= 0),
    CHECK ((status = 'accepted' AND accepted_call_id IS NOT NULL AND rationale IS NOT NULL
            AND length(rationale)>0) OR
           (status = 'abstained' AND accepted_call_id IS NULL AND rationale IS NULL))
) STRICT;

CREATE TABLE allocation (
    ballot_id TEXT NOT NULL REFERENCES ballot(ballot_id),
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    votes INTEGER NOT NULL CHECK (typeof(votes)='integer' AND votes >= 1),
    PRIMARY KEY (ballot_id, candidate_id)
) STRICT;

CREATE TABLE rng_draw (
    draw_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_run(run_id),
    stream_domain TEXT NOT NULL CHECK (stream_domain IN ('tie-break','support-removal')),
    round_index INTEGER NOT NULL CHECK (typeof(round_index)='integer' AND round_index >= 1),
    stream_name TEXT NOT NULL,
    derived_seed BLOB NOT NULL CHECK (typeof(derived_seed)='blob' AND length(derived_seed)=8),
    seed_version TEXT NOT NULL CHECK(seed_version='qv-seed/v1'),
    coordinates_json TEXT NOT NULL CHECK(json_valid(coordinates_json)),
    algorithm_id TEXT NOT NULL CHECK (algorithm_id IN ('pyrandom-randrange/v1')),
    draw_index INTEGER NOT NULL CHECK (typeof(draw_index)='integer' AND draw_index >= 0),
    chosen_candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    UNIQUE (run_id, stream_domain, round_index)
) STRICT;

CREATE TABLE rng_draw_population (
    draw_id TEXT NOT NULL REFERENCES rng_draw(draw_id),
    position INTEGER NOT NULL CHECK (typeof(position)='integer' AND position >= 0),
    candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    PRIMARY KEY (draw_id, position),
    UNIQUE (draw_id, candidate_id)
) STRICT;

CREATE TABLE round_outcome (
    round_id TEXT PRIMARY KEY REFERENCES round(round_id),
    protected_candidate_id TEXT REFERENCES candidate(candidate_id),
    removed_candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    tie_flag INTEGER NOT NULL CHECK (tie_flag IN (0,1)),
    sealed_at TEXT NOT NULL
) STRICT;

CREATE TABLE final_result (
    run_id TEXT PRIMARY KEY REFERENCES experiment_run(run_id),
    winner_candidate_id TEXT NOT NULL REFERENCES candidate(candidate_id),
    completed_at TEXT NOT NULL
) STRICT;

CREATE TRIGGER sample_member_lineage_insert BEFORE INSERT ON candidate_sample_member
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM candidate_sample s JOIN candidate c ON c.candidate_id=NEW.candidate_id
    JOIN candidate_label l ON l.candidate_id=c.candidate_id
      AND l.label_policy_id=s.label_policy_id
    JOIN candidate_presentation p ON p.candidate_id=c.candidate_id AND p.template_id=s.template_id
    WHERE s.sample_id=NEW.sample_id AND c.release_id=s.release_id
  ) THEN RAISE(ABORT,'candidate_sample_member lineage mismatch') END;
END;

CREATE TRIGGER voter_permutation_lineage_insert BEFORE INSERT ON voter_permutation
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM voter v JOIN experiment_run r ON r.run_id=v.run_id
    JOIN matched_set m ON m.matched_set_id=r.matched_set_id
    JOIN experiment_config ec ON ec.config_id=m.config_id
    JOIN experiment_definition ed ON ed.definition_id=ec.definition_id
    JOIN candidate_sample_member sm ON sm.sample_id=ed.sample_id
    WHERE v.voter_id=NEW.voter_id AND sm.candidate_id=NEW.candidate_id
  ) THEN RAISE(ABORT,'voter_permutation sample lineage mismatch') END;
END;

CREATE TRIGGER round_candidate_lineage_insert BEFORE INSERT ON round_candidate
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM round rr JOIN experiment_run r ON r.run_id=rr.run_id
    JOIN matched_set m ON m.matched_set_id=r.matched_set_id
    JOIN experiment_config ec ON ec.config_id=m.config_id
    JOIN experiment_definition ed ON ed.definition_id=ec.definition_id
    JOIN candidate_sample_member sm ON sm.sample_id=ed.sample_id
    WHERE rr.round_id=NEW.round_id AND sm.candidate_id=NEW.candidate_id
  ) THEN RAISE(ABORT,'round_candidate sample lineage mismatch') END;
END;

CREATE TRIGGER turn_same_run_insert BEFORE INSERT ON turn
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM round rr JOIN voter v ON v.voter_id=NEW.voter_id
    WHERE rr.round_id=NEW.round_id AND rr.run_id=v.run_id
  ) THEN RAISE(ABORT,'turn voter and round must share run') END;
END;

CREATE TRIGGER allocation_active_insert BEFORE INSERT ON allocation
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM ballot b JOIN turn t ON t.turn_id=b.turn_id
    JOIN round_candidate rc ON rc.round_id=t.round_id
    WHERE b.ballot_id=NEW.ballot_id AND rc.candidate_id=NEW.candidate_id
  ) THEN RAISE(ABORT,'allocation candidate is not active in turn round') END;
END;

CREATE TRIGGER statement_item_active_insert BEFORE INSERT ON statement_item
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM statement s JOIN turn t ON t.turn_id=s.turn_id
    JOIN round_candidate rc ON rc.round_id=t.round_id
    WHERE s.statement_id=NEW.statement_id AND rc.candidate_id=NEW.candidate_id
  ) THEN RAISE(ABORT,'statement candidate is not active in turn round') END;
END;

CREATE TRIGGER run_status_transition BEFORE UPDATE OF status ON experiment_run
WHEN NOT (
  OLD.status=NEW.status OR
  (OLD.status='created' AND NEW.status='in_progress') OR
  (OLD.status='in_progress' AND NEW.status IN ('paused','complete')) OR
  (OLD.status='paused' AND NEW.status='in_progress')
)
BEGIN SELECT RAISE(ABORT,'illegal experiment_run status transition'); END;

CREATE TRIGGER run_complete_requires_final BEFORE UPDATE OF status ON experiment_run
WHEN NEW.status='complete' AND NOT EXISTS (
  SELECT 1 FROM final_result f WHERE f.run_id=NEW.run_id
)
BEGIN SELECT RAISE(ABORT,'complete run requires final_result in same transaction'); END;

CREATE TRIGGER round_phase_transition BEFORE UPDATE OF phase ON round
WHEN NOT (OLD.phase=NEW.phase OR (OLD.phase='eliciting' AND NEW.phase='sealed'))
BEGIN SELECT RAISE(ABORT,'illegal round phase transition'); END;

CREATE TRIGGER round_seal_requires_outcome BEFORE UPDATE OF phase ON round
WHEN NEW.phase='sealed' AND NOT EXISTS (
  SELECT 1 FROM round_outcome o WHERE o.round_id=NEW.round_id
)
BEGIN SELECT RAISE(ABORT,'sealed round requires round_outcome'); END;

CREATE TRIGGER turn_status_transition BEFORE UPDATE OF status ON turn
WHEN NOT (OLD.status=NEW.status OR (OLD.status='pending' AND NEW.status='committed'))
BEGIN SELECT RAISE(ABORT,'illegal turn status transition'); END;

CREATE TRIGGER sample_status_transition BEFORE UPDATE OF status ON candidate_sample
WHEN NOT (OLD.status=NEW.status OR
  (OLD.status='draft' AND NEW.status='freeze_pending') OR
  (OLD.status='freeze_pending' AND NEW.status='frozen'))
BEGIN SELECT RAISE(ABORT,'illegal candidate_sample status transition'); END;

CREATE TRIGGER model_call_status_transition BEFORE UPDATE OF status ON model_call
WHEN NOT (OLD.status=NEW.status OR
  (OLD.status='started' AND NEW.status IN ('committed','interrupted')))
BEGIN SELECT RAISE(ABORT,'illegal model_call status transition'); END;

CREATE TRIGGER committed_call_complete BEFORE UPDATE OF status ON model_call
WHEN NEW.status='committed' AND (
  NEW.raw_text IS NULL OR NEW.prompt_token_count IS NULL OR
  NEW.completion_token_count IS NULL OR NEW.stop_reason IS NULL OR
  NEW.duration_ms IS NULL OR NEW.diagnostics_json IS NULL OR NEW.committed_at IS NULL
)
BEGIN SELECT RAISE(ABORT,'committed model_call requires complete generation result'); END;

CREATE TRIGGER rng_draw_selected_active BEFORE INSERT ON rng_draw
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM round rr JOIN round_candidate rc ON rc.round_id=rr.round_id
    WHERE rr.run_id=NEW.run_id AND rr.round_index=NEW.round_index
      AND rc.candidate_id=NEW.chosen_candidate_id
  ) THEN RAISE(ABORT,'rng_draw selected candidate is not active in run round') END;
END;

CREATE TRIGGER rng_population_lineage BEFORE INSERT ON rng_draw_population
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM rng_draw d JOIN round rr
      ON rr.run_id=d.run_id AND rr.round_index=d.round_index
    JOIN round_candidate rc ON rc.round_id=rr.round_id
    WHERE d.draw_id=NEW.draw_id AND rc.candidate_id=NEW.candidate_id
  ) THEN RAISE(ABORT,'rng population candidate is not active in run round') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM rng_draw d WHERE d.draw_id=NEW.draw_id
      AND d.draw_index=NEW.position AND d.chosen_candidate_id<>NEW.candidate_id
  ) THEN RAISE(ABORT,'rng selected index does not name selected candidate') END;
END;

CREATE TRIGGER round_outcome_lineage BEFORE INSERT ON round_outcome
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM round_candidate rc
    WHERE rc.round_id=NEW.round_id AND rc.candidate_id=NEW.removed_candidate_id
  ) THEN RAISE(ABORT,'removed candidate is not active in outcome round') END;
  SELECT CASE WHEN NEW.protected_candidate_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM round_candidate rc
    WHERE rc.round_id=NEW.round_id AND rc.candidate_id=NEW.protected_candidate_id
  ) THEN RAISE(ABORT,'protected candidate is not active in outcome round') END;
  SELECT CASE WHEN NEW.protected_candidate_id=NEW.removed_candidate_id
    THEN RAISE(ABORT,'protected and removed candidates must differ') END;
  SELECT CASE WHEN EXISTS (
    SELECT 1 FROM round rr JOIN experiment_run r ON r.run_id=rr.run_id
    WHERE rr.round_id=NEW.round_id AND
      ((r.regime='support' AND NEW.protected_candidate_id IS NULL) OR
       (r.regime='opposition' AND NEW.protected_candidate_id IS NOT NULL))
  ) THEN RAISE(ABORT,'outcome protected candidate conflicts with voting regime') END;
END;

CREATE TRIGGER ballot_call_lineage BEFORE INSERT ON ballot
WHEN NEW.accepted_call_id IS NOT NULL
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM model_call c WHERE c.call_id=NEW.accepted_call_id
      AND c.turn_id=NEW.turn_id AND c.status='committed'
  ) THEN RAISE(ABORT,'accepted ballot call must be committed for same turn') END;
END;

CREATE TRIGGER statement_call_lineage BEFORE INSERT ON statement
WHEN NEW.accepted_call_id IS NOT NULL
BEGIN
  SELECT CASE WHEN NOT EXISTS (
    SELECT 1 FROM model_call c WHERE c.call_id=NEW.accepted_call_id
      AND c.turn_id=NEW.turn_id AND c.status='committed'
  ) THEN RAISE(ABORT,'accepted statement call must be committed for same turn') END;
END;

CREATE TRIGGER frozen_sample_immutable BEFORE UPDATE ON candidate_sample
WHEN OLD.status='frozen'
BEGIN SELECT RAISE(ABORT,'frozen candidate_sample is immutable'); END;

CREATE TRIGGER frozen_sample_member_insert BEFORE INSERT ON candidate_sample_member
WHEN (SELECT status FROM candidate_sample WHERE sample_id=NEW.sample_id)<>'draft'
BEGIN SELECT RAISE(ABORT,'frozen sample members are immutable'); END;

CREATE TRIGGER model_call_identity_immutable BEFORE UPDATE ON model_call
WHEN OLD.turn_id IS NOT NEW.turn_id OR OLD.attempt_index IS NOT NEW.attempt_index OR
     OLD.invocation_index IS NOT NEW.invocation_index OR
     OLD.prompt_messages_json IS NOT NEW.prompt_messages_json OR
     OLD.prompt_sha256 IS NOT NEW.prompt_sha256 OR OLD.seed IS NOT NEW.seed OR
     OLD.status IN ('committed','interrupted')
BEGIN SELECT RAISE(ABORT,'model call identity/terminal row is immutable'); END;
