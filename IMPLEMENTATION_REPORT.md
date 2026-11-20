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
- Template: MEDS_algorithm_template @ `731e759` (temporal-leakage conformance, #52); core `meds-algorithm` @
  `fa8cdfad1140aebf01bd0b544aee6ff580ec98d3` (main with #21, `conformance/temporal.py`; PyPI 0.4.0a5 predates it),
  contract v4.

## Scientific assumptions and decisions

- Mode: **confirmation required** (default; exploratory mode was not enabled).
- Unresolved blocking questions: none block the fixture gates. One MIMIC-IV choice is open: whether to map drug
  names to ATC for 0826 (the 0226 decision kept names whole; NDC exists only in MIMIC-IV-MEDS after 0.1.0).
- Provisional choices still requiring review before a faithful-reproduction claim: the vital window anchoring and
  the stage-2 learning rate (routine, below). This is a MIMIC-IV adaptation of an Epic-Cosmos model and cannot
  reproduce the paper's numbers.

| Missing/conflicting detail | Sources consulted | Options and scientific consequences | Choice/rationale and status | User decision or source resolution | Affected code/tests/results |
|---|---|---|---|---|---|
| Which version to implement | 0226 and 0826 papers | Keep both behind a switch / 0826 only | 0826 only | **Confirmed** by Florent (2026-11-20): "just use the last curiosity version, don't try to make it generic"; pediatric/birth representations must stay supported for other datasets | whole package |
| Which MIMIC-IV codes form each category | 0826 Sec. 2, Table 2; MIMIC-IV-MEDS 0.1.0 and HEAD `event_configs.yaml`; ETL survey (below) | — | Dataset knowledge lives in `datasets/MIMIC-IV.yaml` (ACES plain predicates; `null` = not recorded) | Category-file approach **confirmed** 2026-11-09 for 0226; MIMIC entries below | `datasets/MIMIC-IV.yaml`, `category_predicates` |
| Lab collection marker (0826 Sec. 2.3.1) | MIMIC-IV-MEDS ≥ 0.1.0 (`LAB//SPECIMEN_COLLECTED` at charttime, `LAB//RESULT` at storetime), eICU-MEDS (SPECIMEN_DRAWN/RESULT); NWICU, INSPIRE, AUMCdb, SICdb, HiRID, OMOP: one row, no result time | Synthesize a marker at result − 1 s (paper fallback) / use the ETL's own rows | Use the ETL's value-less lab rows as markers; never synthesize. On single-row ETLs a synthesized marker would duplicate every lab at the same time with no information | Resolved from the ETL survey; reported to Florent 2026-11-20 | `lab` category, `events` |
| Vitals source in MIMIC-IV | MIMIC-IV-MEDS: chartevents (exact times), hosp/omr (00:00 of chartdate; 0.1.0 stores `S/D` text) | chartevents only / + OMR | chartevents non-invasive + arterial BP, weight (kg, lbs), height (cm, in); OMR excluded because its 00:00 time makes a value visible before it was measured that day | **Confirmed** by Florent (2026-11-20): "ok good to keep vitals like this, documented" | `datasets/MIMIC-IV.yaml` |
| Medications without ATC | MIMIC-IV-MEDS: drug names; route/frequency as extra columns; NDC only at HEAD | Keep names whole / NDC→ATC mapping | Names whole (ATC 3/2/rest split applies only to ATC-shaped codes); route and frequency tokens from the ETL columns; no dose column exists | Names-whole **confirmed** 2026-11-09 (0226); attributes per paper | `medication` category, `MEDICATION_ATTRIBUTES` |
| Inclusion thresholds (≥400k numeric-lab events, ≥22k non-numeric, ≥10k procedures/medication attributes/result values) | 0826 Sec. 2.3-2.6, Table 3 totals | Absolute counts (exclude almost everything on MIMIC) / shares | Each count divided by its Table 3 category total, applied as a share of the dataset's training events; the 500-observation minimum for lab bucketing stays absolute (a statistical requirement) | Routine (configurable) | `config.py`, `_Vocabulary.fit` |
| Right-censoring cutoff | 0826 Sec. 2.2.2 (pad to the data cutoff with six-month tokens) | Global latest time / per subject | min(latest training time, last event + `follow_up_days` = 365): MIMIC-IV dates are shifted per patient, and MIMIC-IV records out-of-hospital deaths for one year after discharge | Routine (configurable); per-subject rule asked about by Florent 2026-11-20 | `events` (`censor`), `sequences` |
| Vital downsampling window | 0826 Sec. 2.4 ("latest value in each rolling 6-hour window") | Windows anchored on data / on the clock | Clock-aligned 6-hour windows; a value is hidden only once the later value of its window is visible, so prompts never depend on the future | Routine | `events` (`superseded`), `sequences` |
| Stage 2 learning rate | 0826 Sec. 3.2 (steps and epochs given, schedule not) | — | Constant at the final stage-1 rate (`learning_rate × final_lr_ratio`) | Routine | `_model._extend` |
| Model size | 0826: ~4.3B, 40 layers, 75 tokens/parameter; MIMIC-IV is orders of magnitude smaller | — | Default stays Curiosity-S size (6 layers, 768); `capacity`/`num_layers` scale it | Routine | `config.py` |
| Generation budget | 0826 Sec. 4.1 (60 simulations, 3,000-4,000 tokens, 2-4 retries per task) | — | n = 60, 4,000 tokens, 5 attempts (four retries, the 365-day mortality setting) for a task-independent `infer_trajectories` | Routine (configurable) | `config.py` |
| Key with no resolved future | Contract §5 (finite probabilities) | — | Fallback 0.5, keys in `unresolved_keys.arrow` | **Confirmed** 2026-11-09: "Finite fallback + flag file" | `predict_zero_shot_materialized` |
| Futures that cannot answer the task | 0226 Eq. 2; 0826 Sec. 4.1 (incomplete simulations discarded) | — | Probability = positives / resolved futures | **Confirmed** 2026-11-09 | `_aggregate` |
| Time-token edges, embedding tying, KV heads, optimizer defaults, `[UNK]` | as recorded for 0226 | — | Unchanged (0826: unchanged representations) | Resolved from source / routine | `_tokenizer.py`, `_model.py` |

## Profile choices

### zero_shot_materialized

- Lifecycle justification: the paper's whole method is materialized zero-shot prediction: pretrain a generative
  model on unlabeled timelines, sample futures from a prompt ending at the prediction time, and evaluate each task
  on the futures. That is `preprocess_data → pretrain → infer_trajectories → predict_zero_shot_materialized`.
- Implemented profile methods, algorithm-owned configuration and payload formats: `Curiosity` implements all four
  methods; `CuriosityConfig` extends core `NNConfig`. Payloads: `curiosity-events/v2`, `curiosity-qwen2/v2`,
  `curiosity-trajectories/v2`, `curiosity-predictions/v2`.
- Frozen/fitted parameters and permitted label access: vocabulary, lab buckets, text-result and attribute values,
  vital codes, races, year range and the censoring cutoff are fitted on `train` only. Weights are trained on
  `train`; `tuning` is used only for early stopping. No command reads label values.
- Local and MEDS-DEV evidence for this profile: see Verification.

## Source-element ledger

| Source element | Status: ported/adapted/omitted | Implementation evidence | Justification/deviation |
|---|---|---|---|
| BOS prepended to every sequence (0826 2.2.1) | ported | `sequences` | Packing concatenates BOS-led sequences (no separator) |
| Age at start of timeline: days, weeks, 4-week, 6-month, then 1-year buckets; year-at-start in 5-year buckets (0826 2.1) | ported | `age_token`, `year_token` | Start = first visible event, or the prediction time when none; MIMIC-IV years are shifted per patient (noise, not information) |
| Sex, race tokens (0226 5.2.1) | adapted | `static`, `race` | Race = first recorded value, shown only once recorded |
| Birth event, gestational age (lt22w…gt44w), birth weight with lab buckets (0826 2.1) | ported | `birth`, `gestational_age`, `birth_weight` categories | MIMIC-IV records none (`null`); MEDS_BIRTH there is computed, so it only dates the age token |
| Death at the last instant of its date, preliminary causes, closing token, EOS (0826 2.2.2) | ported | death block in `events` | Later events dropped; MIMIC-IV has no cause of death |
| Right-censored records padded with six-month tokens (0826 2.2.2) | adapted | `censor`, `sequences` | Per-subject cutoff (decision above); padding never enters a prompt |
| Encounters (0226 5.2.2) | adapted | `encounter` | MIMIC codes carry type and unit; no chief complaint |
| Diagnoses ICD 3/2/2 (0226 5.2.3) | ported | `_split` | ICD-9 also split |
| Lab collection marker at collection, result at result time (0826 2.3.1) | adapted | `lab` category | ETL rows used as-is (decision above) |
| Numeric labs: 30 shared percentile buckets, ≥500 values per code (0826 2.3.4) | ported | `LAB_LEVELS`, `quantiles` | Token keeps MIMIC's unit (part of the code) |
| Non-numeric labs: result tokens, Other, missing (0826 2.3.2) | ported | `RESULT//*` | Global result values |
| Vitals: one binned token (Table 2), plausibility bounds, 6-hour downsampling (0826 2.4) | ported | `VITALS`, `VITAL_CATEGORIES` | Units converted from the source category; window anchoring routine |
| Procedures ≥ threshold (0826 2.5) | adapted | `procedure_min_share` | Share instead of count |
| Medications: ATC 3/2/rest, route/form, dose/strength, frequency tokens, reported flag (0826 2.6) | adapted | `medication`, `reported_medication`, `MEDICATION_ATTRIBUTES` | Attribute tokens per value (not per ATC code); MIMIC: names whole, no dose, no reported medications |
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
  before the prediction time: BOS, sex, race once recorded, age and year at the start of the visible timeline, the
  visible events (vitals not yet superseded by then) with time tokens, then the elapsed-gap time tokens. It samples
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
  generable code or value are rejected before any work (`_check_generable`), e.g. chartevents other than the vitals,
  eMAR or infusion codes. Predicates on `other_cols` are rejected.

## Inputs and predicates

- MEDS assumptions: split-sharded MEDS (`data/<split>/*.parquet`) with `subject_id`, `time`, `code`, `numeric_value`;
  optional `text_value` (non-numeric labs, text vitals), `race` (MIMIC-IV-MEDS admission rows), and medication
  columns `route`, `dose`, `frequency`. `train` is required; `tuning` is optional. `MEDS_BIRTH` dates the age token.
- Dataset category files: one ACES plain predicate per category in `config.CATEGORIES` (`curiosity_sex` …
  `curiosity_procedure`; `null` when the dataset does not record it), selected from `additional_predicates_file`,
  else the bundled `datasets/<dataset_name>.yaml` named by the dataset's `metadata/dataset.json`, else an error.
  Only `MIMIC-IV.yaml` is bundled (MIMIC-IV-MEDS 0.1.0 conventions).
- Required predicate names: preprocessing needs the `curiosity_*` predicates only; prediction needs the task's
  criteria and the dataset predicates MEDS-DEV binds.
- Predicate interpretation and derived features: predicates are interpreted only by ACES on real history + generated
  events. Generated numeric values are sampled within their bucket.

## Intermediate artifacts

| Command | Payload format/version | Compatibility check | Inputs/provenance |
|---|---|---|---|
| `preprocess_data` | `curiosity-events/v2`: `vocabulary.json`, `events/<split>/<shard>.parquet` (`subject_id, time, rank, order, tokens, superseded`), `subjects/<split>/<shard>.parquet` (`static, race, race_time, birth, censor`), `payload.json` | — | vocabulary digest, fitted split, category-file route/path/digest, SHA-256 of every MEDS shard, per-split subject/event counts |
| `pretrain` | `curiosity-qwen2/v2`: `model/`, `payload.json` | data payload format | vocabulary digest; per-epoch losses, optimizer steps, stage-2 summary, parameters, device |
| `infer_trajectories` | `curiosity-trajectories/v2`: `events.parquet` (with `text_value`), `samples.parquet`, `payload.json` | data and model formats; equal vocabulary digests | n, sampling policy, horizon, budget, prompt rule, stop reasons |
| `predict_zero_shot_materialized` | `curiosity-predictions/v2`: `predictions.parquet`, `unresolved_keys.arrow`, `relaxations.yaml` | trajectories format and vocabulary; every key has `n_samples` samples; task generability | core records task provenance |

## Scientific deviations

1. Data and vocabulary: Epic Cosmos is replaced by MIMIC-IV codes (no ATC, LOINC, CPT; no chief complaint, births,
   gestational age, birth weight, cause of death or patient-reported medications; OMR excluded pending decision).
2. Thresholds are shares of the dataset's events, not Cosmos counts; the right-censoring cutoff is per subject.
3. Scale: Curiosity-S size, early stopping on tuning loss; stage 2 as specified but at a routine learning rate.
4. Task-independent generation with one horizon and budget; ACES decides when a task resolves.
5. Probability extends discarding of incomplete simulations to futures outside the task population, with a finite
   fallback.

## Verification

### Temporal leakage

- Shared checks (core `fa8cdfa`): `test_end_to_end` reruns the lifecycle on a copy where every evaluated subject's
  events after its latest prediction time are rewritten and requires bit-identical probabilities; the null-world test
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
  before, inside and after a vital window, with race recorded later (`test_prompt_uses_only_history_at_the_prediction_time`).
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
- How the signal enters the algorithm's selected codes/features: both drug codes match `curiosity_medication` and are
  asserted to be vocabulary codes before the behavioral assertion; MEDS_DEATH must appear among generated events.
- Example generator used or adaptation made: `tests/e2e.build_fixture` (MIMIC-IV-MEDS-style codes, ACES labels,
  core split canaries) replaces core `build_signal_dataset`, whose codes are outside the paper's categories.
- Positive-control result (fixture 1000/100/40 subjects, `TEST_CONFIG`, n = 8): mean predicted probability **0.99**
  for signal-drug held-out subjects vs **0.01** for control-drug subjects; the test requires a difference above 0.5.
- Shuffled-label negative-control result (outcome a coin flip): **0.65** vs **0.57**; the test requires |difference|
  below 0.3.

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local (WSL Ubuntu, CPU, 2026-11-20) | `uv run pytest -rs` | 43 passed, 3 skipped (null world: zero-shot; opt-in `meds_dev` and `real_data`) | Conformance (lifecycle, label-blind rerun, future-blind rerun with bit-identical probabilities, split canaries, task binding, ledger), primary behavior, 36 algorithm-specific test cases incl. prompt-level temporal invariance |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | not rerun for 0826 | 0226 gate passed on 2026-11-09 |
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

- Algorithm commit and dirty state: recorded per artifact manifest; branch `curiosity-0826`.
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
- Generated ICD/ATC sub-token sequences can form codes that do not exist (invalid-token counts are recorded).
- `meds-algorithm` is pinned to a git commit until a release contains `conformance/temporal.py`.
- Fixture tests run on CPU with one torch thread for bitwise-reproducible reruns; GPU runs are not bitwise reproducible.
