# Implementation report: Curiosity

## Source

- Paper: Nagarathnam et al., "Curiosity-0826-4B: Generative Medical Event Sequence Modeling Across Pediatric and
  Adult Populations", arXiv:2611.15710 **v1** (2026-11-11), read in full. Its Sec. 2 lists every tokenization change
  from Curiosity-0226-1B and states that "representations not discussed are unchanged from Curiosity-0226-1B".
- Unchanged parts follow Curiosity-0226: Waxler et al., "Generative Medical Event Models Improve with Scale",
  arXiv:2508.12104 **v3** (2025-11-07), Sec. 5 and Appendices A-J. This repository implemented 0226 first
  (commits up to `9299e39`); it now implements 0826 only.
- No code, weights or data were released for either paper (Epic Cosmos is not public): this is a **paper-only**
  port. Every behavior is taken from the papers or is a recorded adaptation.
- Cited source used for an unspecified detail: ETHOS time tokens (Renc et al. 2024, `ipolharvard/ethos-paper`
  `ethos/tokenize/special_tokens.py`, commit a1f03ec), which the 0226 paper says it adapted.
- Reused packages: `aces-relaxations==0.1.0` (PyPI; ACES extraction of the task on history + futures), `es-aces`
  0.7.4, HF `transformers` (`Qwen2ForCausalLM`, the 0226 architecture), `torch`.
- Template: MEDS_algorithm_template @ `731e759` (temporal-leakage conformance, #52); core `meds-algorithm==0.4.0a6` (the first release with `conformance/temporal.py`),
  contract v4.

## Scientific assumptions and decisions

- Mode: **confirmation required** (default; exploratory mode was not enabled).
- Unresolved blocking questions: none.
- Provisional choices still requiring review before a faithful-reproduction claim: the dataset-agnostic tokenization (a deliberate adaptation, below), the
  downsampling window anchoring and the stage-2 learning rate (routine). This is a MIMIC-IV adaptation of an Epic-Cosmos model and cannot
  reproduce the paper's numbers.

| Missing/conflicting detail | Sources consulted | Options and scientific consequences | Choice/rationale and status | User decision or source resolution | Affected code/tests/results |
|---|---|---|---|---|---|
| Which version to implement | 0226 and 0826 papers | Keep both behind a switch / 0826 only | 0826 only | **Confirmed** by Florent (2026-11-20): "just use the last curiosity version, don't try to make it generic"; pediatric/birth representations must stay supported for other datasets (birth tokens later superseded by the dataset-agnostic decision below; pediatric age buckets kept) | whole package |
| Event categories and code hierarchy (dataset-agnostic tokenization) | 0226 Sec. 5.2 and 0826 Sec. 2: a fixed vocabulary of event types (ICD diagnoses split 3/2/2, top numeric labs with 30 buckets, non-numeric labs, Table 2 vitals with clinical bins, ATC medications split 3/2/rest with route/dose/frequency tokens, procedures, births, causes of death); codes outside these types are excluded. Previous port: `curiosity_*` ACES predicates per dataset mapped codes to these types (only MIMIC-IV mapped), then an `other` catch-all | (1) paper categories with a category file per dataset; (2) categories plus `other`; (3) **no categories**: every code a whole token. (3) runs on any MEDS dataset with no per-dataset work and answers any task whose codes occur in training; it loses the code hierarchy (rare diagnoses and drugs no longer share ICD/ATC sub-tokens), clinical vital bins and unit merging (kg and lb stay distinct codes) | (3), as an explicit adaptation: every code one whole-code token (no ICD/ATC split); a code with ≥ `min_bucket_observations` (500) numeric training values gets the 30 shared percentile-bucket tokens (0826 Sec. 2.3.4) — numeric codes need no category; value-less rows of codes with text values get result tokens, rare results `RESULT//[OTHER]`, no result `RESULT//[MISSING]` (0826 Sec. 2.3.2 applied to every code); the optional `route`, `dose`, `frequency` columns give attribute tokens on any row (0826 Sec. 2.6; the ETL decides whether route is a column or part of the code); one frequency threshold for codes, result and attribute values: ≥ max(`min_code_count` = 10, `min_code_share` = 1e-6 × training events); rarer codes become `[UNK]` (as in the paper), rarer results `[OTHER]`, rarer attribute values no token; the latest value of each numeric code in each 6-hour window (0826 Sec. 2.4, applied to every numeric code). Only MEDS standard codes are special: `MEDS_BIRTH` dates the age token, `MEDS_DEATH` is placed at the end of its date and followed by EOS. Category files, `curiosity_*` predicates, Table 2 vital bins, gestational-age and birth-weight tokens, causes of death after death and the patient-reported flag are removed | **Confirmed** by Florent (2026-11-24): "for the numerics we don't need a category, we just do the discretization into bins when meds code has numerics ... we will skip hierarchy decomposition so that it is dataset agnostic. i know this is slightly different from the paper"; text results "yes give result tokens for text values"; route/dose/frequency columns "yes"; threshold "yes but let's do unk to be closer to the paper"; downsampling "keep"; category path "remove". Supersedes "paper categories only" (2026-11-09) and the `other` category (2026-11-24, PR #5 before this change) | `_tokenizer.py`, `config.py` (`min_code_count`, `min_code_share`, `min_bucket_observations`, `downsample_hours`); `test_every_code_is_one_whole_token_with_its_value_tokens`, `test_rare_codes_become_unk_and_rare_values_other`, `test_numeric_codes_are_downsampled_causally` |
| Code hierarchy without categories | 0226 Sec. 5.2.3 (ICD 3/2/2), 0826 Sec. 2.6 (ATC 3/2/rest); the agnostic tokenizer keeps codes whole | No hierarchy (rare codes `[UNK]`) / a per-dataset rule file | Optional `hierarchy_file` (YAML): rules with an ACES-style `code: {regex: ...}`; the matched prefix stays in the head; the rest of the code is split on the rule's `separator` or into character `widths` (remainder as a last part); head `CODE//<prefix><first part>` plus depth-tagged `SUB//<depth>//<part>` tokens, each kept by the frequency threshold; a rare head is `[UNK]`, a rare part truncates the code to its known ancestor; generation rejoins the parts exactly; first matching rule wins; codes no rule matches stay whole (units are never split). Rules are frozen into `vocabulary.json`. Shipped example: `hierarchies/MIMIC-IV.yaml` (ICD-9/10 diagnoses 3/2/2, both MIMIC-IV-MEDS layouts), not used unless selected | **Confirmed** by Florent (2026-11-24): "specify a yaml file with a predicate for codes to split as hierarchy using '/'"; separator "per rule, after the matched prefix"; widths "yes"; head and depth-tagged sub-tokens rejoined on generation "yes"; "just an example file shipped in the package, could be for mimic" | `load_hierarchy`, `split_code`, `_Vocabulary._code_tokens/decode`; `test_split_code_after_the_matched_prefix`, `test_hierarchy_files_are_validated`, `test_hierarchy_tokens_share_ancestors_truncate_rare_parts_and_decode_exactly` |
| Lab collection marker (0826 Sec. 2.3.1) | MIMIC-IV-MEDS ≥ 0.1.0 (`LAB//SPECIMEN_COLLECTED` at charttime, `LAB//RESULT` at storetime), eICU-MEDS (SPECIMEN_DRAWN/RESULT); NWICU, INSPIRE, AUMCdb, SICdb, HiRID, OMOP: one row, no result time | Synthesize a marker at result − 1 s (paper fallback) / use the ETL's own rows | The ETL's own value-less rows are ordinary code tokens; nothing is synthesized (a synthesized marker on single-row ETLs would duplicate every lab at the same time) | Resolved from the ETL survey; reported to Florent 2026-11-20 | `events` |
| Right-censoring cutoff | 0826 Sec. 2.2.2 (pad to the data cutoff with six-month tokens) | Global latest time / per subject | min(latest training time, last event + `follow_up_days` = 365): MIMIC-IV dates are shifted per patient, and MIMIC-IV records out-of-hospital deaths for one year after discharge | Routine (configurable); per-subject rule asked about by Florent 2026-11-20 | `events` (`censor`), `sequences` |
| Downsampling window | 0826 Sec. 2.4 ("latest value in each rolling 6-hour window", for vitals) | Windows anchored on data / on the clock | Clock-aligned 6-hour windows per numeric code; a value is hidden only once the later value of its window is visible, so prompts never depend on the future | Routine (`downsample_hours`) | `_supersede`, `sequences` |
| Stage 2 learning rate | 0826 Sec. 3.2 (steps and epochs given, schedule not) | — | Constant at the final stage-1 rate (`learning_rate × final_lr_ratio`) | Routine | `_model._extend` |
| Model size | 0226 Table 3 (S 6×768, M 12×768, L 16×2048; 12/12/32 heads; MLP 4×); 0826 Sec. 3.1 (40 layers, ~4.3B, width not reported) | — | `size` presets: small/medium/large = 0226 S/M/L (62M, 119M, 1.09B measured at the 7,105-token vocabulary); xlarge = 40 layers × 2,560 (40 heads), the width that gives ~4.2B at 40 layers; default `small` (MIMIC-IV is orders of magnitude below the papers' budgets). `num_layers`/`reference_heads`/`capacity` override a preset | Resolved from source (S/M/L); xlarge width routine | `config.SIZES`, `test_size_presets_follow_the_papers` |
| Generation budget | 0826 Sec. 4.1 (60 simulations, 3,000-4,000 tokens, 2-4 retries per task) | — | n = 60, 4,000 tokens, 5 attempts (four retries, the 365-day mortality setting) for a task-independent `infer_trajectories` | Routine (configurable) | `config.py` |
| Key with no resolved future | Contract §5 (finite probabilities) | — | Fallback 0.5, keys in `unresolved_keys.arrow` | **Confirmed** 2026-11-09: "Finite fallback + flag file" | `predict_zero_shot_materialized` |
| Futures that cannot answer the task | 0226 Eq. 2; 0826 Sec. 4.1 (incomplete simulations discarded) | — | Probability = positives / resolved futures | **Confirmed** 2026-11-09 | `_aggregate` |
| Time-token edges, embedding tying, KV heads, optimizer defaults | as recorded for 0226 | — | Unchanged (0826: unchanged representations) | Resolved from source / routine | `_tokenizer.py`, `_model.py` |

## Profile choices

### zero_shot_materialized

- Lifecycle justification: the paper's whole method is materialized zero-shot prediction: pretrain a generative
  model on unlabeled timelines, sample futures from a prompt ending at the prediction time, and evaluate each task
  on the futures. That is `preprocess_data → pretrain → infer_trajectories → predict_zero_shot_materialized`.
- Implemented profile methods, algorithm-owned configuration and payload formats: `Curiosity` implements all four
  methods; `CuriosityConfig` extends core `NNConfig`. Payloads: `curiosity-events/v2`, `curiosity-qwen2/v2`,
  `curiosity-trajectories/v2`, `curiosity-predictions/v2`.
- Frozen/fitted parameters and permitted label access: vocabulary, value buckets, text-result and attribute
  values, races, year range and the censoring cutoff are fitted on `train` only. Weights are trained on
  `train`; `tuning` is used only for early stopping. No command reads label values.
- Local and MEDS-DEV evidence for this profile: see Verification.

## Source-element ledger

| Source element | Status: ported/adapted/omitted | Implementation evidence | Justification/deviation |
|---|---|---|---|
| BOS prepended to every sequence (0826 2.2.1) | ported | `sequences` | Packing concatenates BOS-led sequences (no separator) |
| Age at start of timeline: days, weeks, 4-week, 6-month, then 1-year buckets; year-at-start in 5-year buckets (0826 2.1) | ported | `age_token`, `year_token` | Start = first visible event, or the prediction time when none; MIMIC-IV years are shifted per patient (noise, not information) |
| Sex, race tokens (0226 5.2.1) | adapted | `static`, `_races` | Static codes (time null, e.g. sex) form the header; a race column (MIMIC-IV-MEDS 0.1.0) gives a race header token shown once recorded; race recorded as codes (0.4.0 `RACE//`) is an ordinary event |
| Birth event, gestational age, birth weight (0826 2.1) | adapted | whole codes and value buckets | No special tokens (agnostic decision); `MEDS_BIRTH` dates the age token |
| Death at the last instant of its date, EOS (0826 2.2.2) | ported | death block in `events` | `MEDS_DEATH`; later events dropped; preliminary causes of death are ordinary codes |
| Right-censored records padded with six-month tokens (0826 2.2.2) | adapted | `censor`, `sequences` | Per-subject cutoff (decision above); padding never enters a prompt |
| Encounters (0226 5.2.2) | adapted | whole codes | MIMIC codes carry type and unit; no chief complaint |
| Diagnoses ICD 3/2/2 (0226 5.2.3) | adapted | optional `hierarchy_file`; example `hierarchies/MIMIC-IV.yaml` | Off by default (whole codes); a hierarchy file restores the split for the codes its rules match |
| Lab collection marker at collection, result at result time (0826 2.3.1) | adapted | whole codes | ETL rows used as-is (decision above) |
| Numeric labs: 30 shared percentile buckets, ≥500 values per code (0826 2.3.4) | adapted | `LAB_LEVELS`, `quantiles` | Applied to every numeric code; the token keeps the code's unit |
| Non-numeric labs: result tokens, Other, missing (0826 2.3.2) | adapted | `RESULT//*` | Applied to every code with text values; global result values |
| Vitals: one binned token (Table 2), plausibility bounds, 6-hour downsampling (0826 2.4) | adapted | value buckets, `_supersede` | Percentile buckets instead of Table 2 bins; downsampling applied to every numeric code; no unit conversion |
| Procedures, labs, medications ≥ count thresholds (0826 2.3-2.6) | adapted | `min_code_count`, `min_code_share` | One threshold for every code; rarer codes `[UNK]` |
| Medications: ATC 3/2/rest, route/form, dose/strength, frequency tokens, reported flag (0826 2.6) | adapted | `ATTRIBUTES` | ATC split only through a hierarchy rule (`widths: [3, 2]`); attribute tokens from the optional columns on any row; no reported flag |
| ETHOS time tokens; geometric-mean time decoding (0226 5.2.7, 5.5) | ported | `TIME_BUCKETS`, `_token_seconds` | Unchanged in 0826 |
| Qwen2 decoder, random init, tied embeddings (0226 5.3.1) | ported | `build_model` | 0826 does not name its architecture |
| Two-stage context 8,192 → 16,384, unscaled RoPE (0826 3, 3.2) | ported | `_extend`, `generation_context` | Stage-2 learning rate routine |
| IsoFLOP scaling / 4.3B size (0826 3.1) | omitted | — | Scaling study; MIMIC-IV is far below the budget |
| Monte Carlo futures, temperature 1, left-truncated prompts (0226 5.5) | ported | `generate` | Budget per 0826 Sec. 4.1 |
| Incomplete simulations discarded (0826 4.1.1) | adapted | `_aggregate` | Positives / resolved futures |
| Task outcome definitions | adapted | ACES via `aces-relaxations.extract` | MEDS tasks are given as ACES criteria |
| Lab-value regression, length of stay, calibration analyses (0826 4.2) | omitted | — | The contract evaluates binary task probabilities |

## Zero-shot materialization

- What `infer_trajectories` generates (samples per row, generation horizon, stopping rule): for every requested
  `(subject_id, prediction_time)`, `_Vocabulary.sequences` builds the prompt from the prepared events recorded at or
  before the prediction time: BOS, static codes, race once recorded, age and year at the start of the visible
  timeline, the visible events (numeric values not yet superseded by then) with time tokens, then the elapsed-gap time tokens. It samples
  `n_samples` = 60 futures at temperature 1, no top-k/top-p, with a per-key RNG seeded from
  `(seed, subject_id, prediction_time)`. Each future runs up to 5 attempts × 4,000 tokens and stops at EOS
  (`death`; EOS follows only a death), BOS (`end_of_record`), generated time beyond prediction_time + 365 d
  (`horizon`), or the token budget (`token_budget`).
- How `predict_zero_shot_materialized` evaluates the task on history plus futures, and aggregates: each (key,
  sample) becomes a pseudo-subject made of the real MEDS events with time ≤ prediction_time plus that sample's
  generated events; the original ACES task runs on them (`aces_relaxations.extract`). A future is resolved if
  positive, or negative with its label window ending by its `end_time` (or in death). Probability = positives /
  resolved; a key with no resolved future gets 0.5 and is listed in `unresolved_keys.arrow`.
- Ledger per evaluated task (sampling approximation; horizon or unresolvable-future effects): `relaxations.yaml`
  (`exact: false`): Monte Carlo, generation limit, task population, fallback.
- Tasks that cannot be evaluated on generated data, and why: tasks whose post-prediction predicates match no
  generable code or value are rejected before any work (`_check_generable`): codes absent from, or rarer than the
  threshold in, the training split. Predicates on `other_cols` are rejected.

## Inputs and predicates

- MEDS assumptions: split-sharded MEDS (`data/<split>/*.parquet`) with `subject_id`, `time`, `code`, `numeric_value`;
  optional `text_value` (result tokens), `race` (MIMIC-IV-MEDS 0.1.0 admission rows), and `route`, `dose`,
  `frequency` (attribute tokens). `train` is required; `tuning` is optional. `MEDS_BIRTH` dates the age token and
  `MEDS_DEATH` closes the record. No dataset-specific file or mapping is needed.
- Required predicate names: preprocessing needs none; prediction needs the task's criteria and the dataset
  predicates MEDS-DEV binds.
- Predicate interpretation and derived features: predicates are interpreted only by ACES on real history + generated
  events. Generated numeric values are sampled within their bucket.

## Intermediate artifacts

| Command | Payload format/version | Compatibility check | Inputs/provenance |
|---|---|---|---|
| `preprocess_data` | `curiosity-events/v2`: `vocabulary.json`, `events/<split>/<shard>.parquet` (`subject_id, time, order, tokens, superseded`), `subjects/<split>/<shard>.parquet` (`static, race, race_time, birth, censor`), `payload.json` | — | vocabulary digest, fitted split, SHA-256 of every MEDS shard, per-split subject/event counts |
| `pretrain` | `curiosity-qwen2/v2`: `model/`, `payload.json` | data payload format | vocabulary digest; per-epoch losses, optimizer steps, stage-2 summary, parameters, device |
| `infer_trajectories` | `curiosity-trajectories/v2`: `events.parquet` (with `text_value`), `samples.parquet`, `payload.json` | data and model formats; equal vocabulary digests | n, sampling policy, horizon, budget, prompt rule, stop reasons |
| `predict_zero_shot_materialized` | `curiosity-predictions/v2`: `predictions.parquet` (keys, `predicted_boolean_value` = probability ≥ 0.5, `predicted_boolean_probability`), `unresolved_keys.arrow`, `relaxations.yaml` | trajectories format and vocabulary; every key has `n_samples` samples; task generability | core records task provenance |

## Scientific deviations

1. Vocabulary (dataset-agnostic, decision above): every MEDS code is a whole-code token with value, result and
   attribute tokens; no event categories, no ICD/ATC hierarchy split, no Table 2 vital bins, no birth, cause-of-death
   or patient-reported special tokens. Data: Epic Cosmos is replaced by MEDS datasets (MIMIC-IV in the tests).
2. One frequency threshold for every code (rarer codes `[UNK]`) instead of per-type Cosmos counts; downsampling of
   every numeric code; the right-censoring cutoff is per subject.
3. Scale: Curiosity-S size, early stopping on tuning loss; stage 2 as specified but at a routine learning rate.
4. Task-independent generation with one horizon and budget; ACES decides when a task resolves.
5. Probability extends discarding of incomplete simulations to futures outside the task population, with a finite
   fallback.

## Verification

### Temporal leakage

- Shared checks (core 0.4.0a6): `test_end_to_end` reruns the lifecycle on a copy where every evaluated subject's
  events after its latest prediction time are rewritten and requires probabilities identical up to rounding (relative 1e-12); the null-world test
  is skipped for zero-shot profiles.
- Finding on the previous (0226) implementation: race came from the first race recorded anywhere in the record and
  age from the first kept event, even when after the prediction time. These were **errors of the MEDS adaptation,
  not of the paper**: 0226 treats sex, race and age as a static demographic header, which is safe when demographics
  are patient-level attributes (Epic) and prompts start from an existing timeline, but MIMIC-IV-MEDS records race on
  admission rows (timed), and a MEDS prediction time can precede every kept event. The port read both from the
  whole record instead of from history at the prediction time. With fixture survivors whose race is recorded
  only at a later readmission, **6 of 40 held-out prompts changed** under `perturb_future`, yet the shared
  prediction-level check **passed**: a probability estimated from 8 futures takes few values, so changed prompts can
  leave it unchanged. The fixture keeps these late-race subjects.
- Algorithm-specific checks added (`tests/test_curiosity.py`): prompts are identical under `perturb_future` on the
  fixture (`test_prompts_are_invariant_to_rewritten_futures`); random future events after random cutoffs never change
  a prompt across every representation (`test_random_futures_never_change_random_prompts`); prompt contents at cutoffs
  before, inside and after a numeric code's 6-hour window, with race recorded later
  (`test_prompt_uses_only_history_at_the_prediction_time`), and the downsampling itself
  (`test_numeric_codes_are_downsampled_causally`).
- Paper rules that would leak if applied to the full record and then truncated (the papers do not say how prompts
  are cut): 0826 vital downsampling (a value is dropped when a later value of its window exists), right-censoring
  padding (its length depends on when the record ends) and EOS. Here they are evaluated at the cutoff instead
  (`superseded`, padding and EOS only in training sequences), which the tests above cover.
- Review of the remaining paths: the vocabulary, buckets and cutoff are fitted on `train` only; pretraining reads
  `train`/`tuning` only; padding and EOS never enter prompts; generation is causal (Qwen2) and seeded per key;
  prediction joins only real events with time ≤ prediction_time. Data-level caveat outside the algorithm: MIMIC-IV-MEDS
  times `dod` deaths and OMR rows at 00:00 of their date.

### Primary behavior

- Signal and expected behavior: in the synthetic ICU fixture, a medication ordered one hour after ICU admission
  (`MEDICATION//START//SIGNAL_DRUG` vs `CONTROL_DRUG`) decides in-ICU death vs discharge 60-90 h after admission.
  The model is pretrained without labels; its futures must carry death after the signal drug.
- How the signal enters the algorithm's selected codes/features: both drug codes are whole-code tokens and are
  asserted to be vocabulary codes before the behavioral assertion; MEDS_DEATH must appear among generated events.
- Example generator used or adaptation made: `tests/e2e.build_fixture` (MIMIC-IV-MEDS-style codes, ACES labels,
  core split canaries) replaces core `build_signal_dataset`, as a zero-shot task needs a temporal structure.
- Positive-control result (fixture 1000/100/40 subjects, `TEST_CONFIG`, n = 8): mean predicted probability **0.99**
  for signal-drug held-out subjects vs **0.01** for control-drug subjects; the test requires a difference above 0.5.
- Shuffled-label negative-control result (outcome a coin flip): **0.65** vs **0.57**; the test requires |difference|
  below 0.3.

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local (WSL Ubuntu, CPU, 2026-11-24, core 0.4.0a6, dataset-agnostic tokenizer) | `uv run pytest -rs` | 47 passed, 3 skipped (null world: zero-shot; opt-in `meds_dev` and `real_data`) | Conformance (lifecycle, label-blind rerun, future-blind rerun with probabilities identical up to rounding, split canaries, task binding, ledger), primary behavior, algorithm-specific tests incl. whole-code tokens with value/result/attribute tokens, `[UNK]` threshold, causal downsampling, prompt-level temporal invariance and size presets |
| Release | `python scripts/check_release.py`; `bash scripts/check_distribution.sh` | both pass | metadata, license, no direct-URL dependencies, installed wheel (plain and orchestration) |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | not rerun for 0826 | 0226 gate passed on 2026-11-09 |
| MEDS-DEV integration (draft `curiosity/small`, MIMIC-IV demo, hypotension task; 0.1.1 wheel with the earlier `other` category) | MEDS-DEV `pytest -m integration --test_model=curiosity/small` | 7 passed, 1 error | Dataset build, task, preprocessing, pretraining, trajectories and predictions ran; evaluation failed because predictions lacked `predicted_boolean_value` (also true of 0.1.0): fixed, written as probability ≥ 0.5 |
| MEDS-DEV integration (draft `curiosity/small`, dataset-agnostic 0.1.1 wheel, demo config) | MEDS-DEV `pytest -m integration --test_model=curiosity/small --test_task=abnormal_lab/vital/hypotension/first_24h --test_dataset=<MIMIC-IV|eICU>` | MIMIC-IV: 9 passed (8 min); eICU: 9 passed (24 min) | Dataset build, task, all four steps and meds-evaluation, on CPU; eICU runs with no dataset-specific configuration, and the hypotension task (mean arterial pressure) is generated and evaluated |
| MIMIC-IV demo | `uv run pytest -m real_data -rs` | not run: awaiting human approval | — |
| Full target dataset | not run | — | Blocked: no evaluation task bundle supplied |

## Review gates

| Gate | Evidence reviewed | Reviewer/approval | Date or reference |
|---|---|---|---|
| Before MIMIC-IV demo | Source fidelity, local/MEDS-DEV results, predicates, demo plan | Requested from Florent; not yet approved | 2026-11-20 |
| Before full target dataset | Demo result, target-dataset/task/predicate compatibility, access, full-run resources | Not reached | — |

## Resource measurements

| Environment/command | Wall seconds | Peak tree PSS bytes | Peak tree RSS bytes | GPU bytes |
|---|---:|---:|---:|---:|
| Fixture lifecycle (TEST_CONFIG, CPU) | recorded per step in each artifact manifest | per manifest | per manifest | null (CPU) |

## Environment and provenance

- Algorithm commit and dirty state: recorded per artifact manifest.
- Dataset/task identifiers: synthetic fixture only; MIMIC-IV demo pending approval.
- Dataset, task, and predicate revisions: MEDS-DEV pinned by core for MEDS-DEV and demo tests.
- Task-bundle kind (clinical ACES tasks / sampled code-by-horizon tasks) and the commit of the tool that built it:
  none supplied.
- Shared task-bundle identifier/path: none supplied.
- Digest of the bundle's task definitions: none supplied.
- Cluster checkout commit and job IDs: none.
- Structured-result branch/PR: none.
- Dependency/platform/hardware summary: Python 3.12 (WSL Ubuntu), torch, transformers 5.22, polars 1.44,
  aces-relaxations 0.1.0, es-aces 0.7.4. Native Windows cannot run ACES extraction: aces-relaxations reopens an open
  `NamedTemporaryFile` (WinError 32).

## Known limitations

- Not a reproduction of the paper's results: different data, vocabulary and scale.
- The shared future-blind rerun compares probabilities only; with few samples it can miss prompt-level leaks, hence
  the prompt-level tests above.
- Only the label window, not criteria windows, is checked against the generation end when resolving a negative.
- Generation runs one key at a time (its n samples batched); prediction duplicates each key's history n times.
- Without a hierarchy file, rare diagnoses and drugs below the threshold are `[UNK]` and cannot be generated; with one,
  a rare code is generated as its known ancestor (e.g. E11 for a rare E11.78).
- Fixture tests run on CPU with one torch thread for bitwise-reproducible reruns; GPU runs are not bitwise reproducible.
