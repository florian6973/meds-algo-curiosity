# Implementation report: Curiosity

## Source

- Paper: Waxler, Blazek, White, Sneider et al., "Generative Medical Event Models Improve with Scale",
  arXiv:2508.12104. The model was named CoMET in v1 (2025-08-16), Comet in v2 (2025-09-23) and Curiosity in v3.
- Exact revision/version: **v3** (2025-11-07), read in full (Sections 1-5, Appendices A-J). The abstract and
  Table 1 match v1. No code, weights or data were released, and the data (Epic Cosmos) is not public, so this
  is a **paper-only** port: there is no source package to depend on. Every behavior below is either taken from
  the paper or an explicitly recorded adaptation.
- Cited source used for an unspecified detail: ETHOS time tokens (Renc et al. 2024, `ipolharvard/ethos-paper`
  `ethos/tokenize/special_tokens.py`, commit a1f03ec), which the paper says it adapted.
- Reused packages: `aces-relaxations` @ `adf435787169047c6e0de662461b4c0d9932608e` (ACES extraction of the task on
  history + futures), `es-aces` 0.7.4, HF `transformers` (`Qwen2ForCausalLM`, the paper's architecture), `torch`.
- Template: MEDS_algorithm_template @ `6e8f557` (v0.4.0a2-3); core `meds-algorithm==0.4.0a5`, contract v4.

## Scientific assumptions and decisions

- Mode: **confirmation required** (default; exploratory mode was not enabled).
- Unresolved blocking questions: none. All material choices below were decided by Florent on 2026-11-09.
- Provisional choices still requiring review before a faithful-reproduction claim: none are provisional. The
  routine defaults listed below are flagged for review in the pre-demo gate. Note that this is a MIMIC-IV
  adaptation of an Epic-Cosmos model, so it cannot reproduce the paper's numbers.

| Missing/conflicting detail | Sources consulted | Options and scientific consequences | Choice/rationale and status | User decision or source resolution | Affected code/tests/results |
|---|---|---|---|---|---|
| Which MIMIC-IV codes form the vocabulary (Epic vocabularies ICD-10-CM/ATC/LOINC/CPT do not exist in MIMIC MEDS) | Paper Sec. 5.2, 5.2.8, Table 2; MIMIC-IV-MEDS 0.1.0 `event_configs.yaml`; MEDS-DEV MIMIC predicates | All train codes (any predicate generable, deviates from exclusions) / paper categories only (tasks on excluded codes unanswerable) / paper + NDC→ATC/ICD-9→10/LOINC remap (lossy back-mapping) | Paper categories only, mapped to MIMIC codes (`MIMIC_IV_CATEGORIES` in `config.py`); tasks whose post-prediction predicates match no vocabulary code are rejected | **Confirmed** by Florent: "Paper categories only" | `config.py`, `_tokenizer.py`, `_check_generable`; `test_fit_keeps_paper_categories_and_train_deciles`, `test_tasks_needing_ungenerable_codes_are_rejected` |
| Applying the token scheme to MIMIC codes (no ATC; ICD-9 and ICD-10 mixed; shifted dates) | Paper Sec. 5.2.1-5.2.5 | Split ICD only / no splitting / also keep years-since-1970 | Split ICD-9 and ICD-10 diagnoses into 3/2/2-character sub-tokens; other codes whole; drop years-since-1970 (MIMIC dates are shifted per patient) | **Confirmed**: "Split ICD only, rest whole" | `_tokenizer.py`; `test_vectorized_transform_matches_scalar_gaps_and_roundtrips` |
| Model size and training budget (paper: 62M-1B params, ~1,000 tokens/param, unreachable on MIMIC) | Paper Sec. 5.3.1, Tables 1 and 3 | S / M / L reference; fixed steps vs early stopping | capacity 1.0 = Curiosity-S (6 layers, 768, 12 heads, MLP 3072, context 8,192); width scales with capacity; up to `max_epochs` with early stopping on the tuning-split next-token loss | **Confirmed**: "S + tuning early stop" | `config.py`, `_model.py`; `test_capacity_and_dropout_control_the_qwen2_architecture` |
| Samples per key and generation stopping for a task-independent `infer_trajectories` | Paper Sec. 2.7.2, 5.5, 5.6 (n = 20-80; 2,000 tokens; "retried up to two times"; temperature 1) | n = 20 / 64 / 80 | n = 20, temperature 1, no top-k/top-p, up to 3 attempts × 2,000 new tokens, stop at MEDS_DEATH, separator, or prediction_time + 365 d; prompt = real events with time ≤ prediction_time | **Confirmed**: "n = 20"; the prompt cutoff and stopping rules were in the same confirmed question | `config.py`, `_model.generate`, `_prompt`; `test_generation_stops_at_horizon_or_stop_token`, `test_prompt_stops_at_prediction_time_and_states_the_elapsed_gap` |
| Futures that cannot answer the task: generation ends before the label window closes (paper Eq. 2), or the future violates a post-prediction criterion (absent from paper) | Paper Eq. 2, Sec. 5.6.1/5.6.3/5.6.4 (discard vs retry vs right-censor conflict) | Drop both / drop (1) and count (2) negative / count both negative | Probability = positives / resolved futures (Eq. 2 extended to futures outside the task population) | **Confirmed**: "pos / (futures that answer the task)" | `_aggregate`; `test_aggregate_extends_equation_2_with_fallback` |
| Key with no resolved future (user first asked for NaN; core 0.4.0a5 rejects non-finite probabilities and has no exclusion policy) | Contract §5; `meds_algorithm/evaluation/predictions.py` | Finite fallback + flag file / fallback 0 / change core | Fallback 0.5, keys listed in `unresolved_keys.arrow`, count recorded in `relaxations.yaml` | **Confirmed**: "Finite fallback + flag file" | `predict_zero_shot_materialized`, `config.fallback_probability` |
| Time-token bucket edges (paper: "1-5 minutes" to "3-6 months", 13 tokens; edges not listed) | Paper Sec. 5.2.7, Table 2; ETHOS `special_tokens.py` (13 tokens `5m-15m`…`3mt-6mt`, `=6mt`) | ETHOS list with a 1-5 min bucket prepended | ETHOS edges (30-day months) plus `1m-5m`: 13 range tokens plus the "6 month" token. The paper's Table 2 count (13) is one fewer; we assume Table 2 does not count the 6-month token separately. Gaps < 1 min get no token; ≥ 180 d give round(gap/180 d) "6 month" tokens; decoding uses the geometric mean (paper Sec. 5.5) and 180 d for "6 month" | Resolved from the cited source (ETHOS); routine for the count discrepancy | `_tokenizer.py`; `test_gap_tokens_follow_ethos_buckets` |
| Embedding tying, KV heads, biases | Paper Table 1 parameter counts; Qwen2 defaults | Tied/untied embeddings; GQA ratio | Tied embeddings: they reproduce Table 1 (62M for S at vocab 7,105; 119M for M), untied do not; KV heads = attention heads (GQA ratio unspecified) | Resolved from source (parameter counts); routine for KV heads | `_model.build_model`; parameter-count test |
| Optimizer details | Paper Sec. 5.3.1 (AdamW, cosine with 10x decay, batch 512 sequences) | — | Routine defaults: peak LR 6e-4, weight decay 0.1, 1% linear warmup, gradient clip 1.0, bf16 autocast on GPU, AdamW betas (0.9, 0.999), attention dropout 0, microbatch 8 × accumulation to 512 sequences, `max_epochs` 20, patience 2 | Routine default (configurable) | `config.py`, `_model.pretrain`; `test_optimizer_and_batching_settings_reach_training` |
| Out-of-vocabulary codes inside a kept category; labs beyond the top 1,000 | Paper Table 2 lists an unknown token | Drop vs `[UNK]` | `[UNK]` token, no decile | Routine | `_tokenizer.transform` |
| Generated events at the prediction time / prompt ending between events | Paper prompts end at an event; MEDS prediction times need not | — | The prompt ends with the real elapsed gap tokens (time ≤ prediction_time has no further real events); generated events start the clock at prediction_time and lie strictly after it (+1 µs if no time token precedes them) | Routine (temporal-isolation consequence) | `_prompt`, `_Vocabulary.decode` |
| Retry semantics and context overflow | Paper Sec. 5.6.4 ("retried up to two times"), Sec. 5.5 (left truncation) | Restart vs continue | Continue: each attempt re-prompts with the latest `context_length − max_new_tokens` tokens of prompt + generation | Routine | `_model.generate` |
| Race token values | Paper: 7 race tokens, categories unlisted | Map to 7 groups vs raw values | Train-seen raw values of MIMIC's `race` column (first non-null per subject); absent column → no race token | Routine | `_tokenizer`, `Curiosity._race` |

## Profile choices

### zero_shot_materialized

- Lifecycle justification: the paper's whole method is materialized zero-shot prediction. It pretrains a
  generative model on unlabeled timelines (Sec. 5.3), samples future timelines from a prompt ending at the
  prediction time (Sec. 5.5), and evaluates each task on the generated futures as the fraction of futures with
  the outcome (Eq. 2). That is `preprocess_data → pretrain → infer_trajectories → predict_zero_shot_materialized`.
  No supervised, finetune or probe use is part of Curiosity's claim; the paper's supervised models are
  baselines, so they were not added.
- Implemented profile methods, algorithm-owned configuration and payload formats: `Curiosity` implements all four
  methods. `CuriosityConfig` extends core `NNConfig`. Payloads are `curiosity-tokens/v1`, `curiosity-qwen2/v1`,
  `curiosity-trajectories/v1` and `curiosity-predictions/v1` (table below).
- Frozen/fitted parameters and permitted label access: the vocabulary, deciles and race values are fitted on
  `train` only. Weights are trained on `train` sequences; the `tuning` loss is used only for early stopping. No
  command reads label values: inference and prediction read keys through `PredictionIndex.read()`
  (`read_index`), and the runner alone attaches truth after scoring.
- Local and MEDS-DEV evidence for this profile: see Verification.

## Source-element ledger

| Source element | Status: ported/adapted/omitted | Implementation evidence | Justification/deviation |
|---|---|---|---|
| Chronological event tokenization with an a-priori vocabulary (Sec. 5.2) | adapted | `_tokenizer._Vocabulary` | Vocabulary built from MIMIC codes of the paper's categories (decision above) |
| Demographics: sex, race, age in 5-year buckets, then BOS (Sec. 5.2.1) | adapted | prefix in `transform` | Age at first kept event; years-since-1970 omitted (shifted MIMIC dates) |
| Encounter start/end tokens with type, department specialty, chief complaint (Sec. 5.2.2) | adapted | `encounter` category | MIMIC admission/discharge/ICU/ED/transfer codes already carry type and location/care unit; MIMIC-IV MEDS has no chief complaint |
| Diagnoses as ICD 3/2/2 sub-tokens (Sec. 5.2.3) | ported | `_split_icd`, `decode` | ICD-9 also split; diagnosis time is the MIMIC ETL's (discharge time) |
| Top 1,000 numeric labs + 10 shared decile tokens (Sec. 5.2.4) | ported | `lab` category, `quantiles` | `LAB//RESULT//*` hosp labs, train deciles per code |
| Medication orders as ATC 3/2/2 (Sec. 5.2.5) | adapted | `medication` category | Pharmacy order codes kept whole (no ATC in MIMIC) |
| Top 1,500 billed procedures (Sec. 5.2.6) | adapted | `procedure` category | ICD procedures, HCPCS, and ICU procedure starts, pooled top 1,500 |
| Exclusions: notes, vitals, administrations, free text (Sec. 5.2.8) | ported | category regexes | chartevents, eMAR, infusions, outputs, OMR, DRG are excluded |
| Death | adapted | `death` category, stop token | Not a token type in the paper; required by MIMIC mortality tasks (confirmed category list) |
| ETHOS time tokens; geometric-mean time decoding (Sec. 5.2.7, 5.5) | ported | `TIME_BUCKETS`, `_token_seconds` | Bucket edges from ETHOS (see decision) |
| Qwen2 decoder, random init, pre-LN/SwiGLU/RoPE (Sec. 5.3.1) | ported | `_model.build_model` | Tied embeddings derived from Table 1 |
| Sizes S/M/L (Table 3) | adapted | `capacity` | S is the reference; M/L need config changes (depth fixed by `num_layers`) |
| Dense packing with separator, context 8,192, batch 512 (Sec. 5.3.1) | ported | `_model._pack`, `sequences_per_step` | Microbatches + accumulation |
| AdamW, cosine with 10x decay (Sec. 5.3.1) | ported | `_model.pretrain` | Other hyperparameters routine |
| Compute-optimal token budgets / scaling laws (Sec. 2.6, 3.1) | omitted | — | Scaling-law study, not part of the algorithm; MIMIC is far below the budget |
| Left-truncated prompt at prediction time; temperature-1 Monte Carlo sampling (Sec. 5.5) | ported | `_prompt`, `_model.generate` | Prompt ends with the elapsed real gap |
| 2,000-token generations, up to 2 retries (Sec. 5.6) | ported | `max_new_tokens`, `max_attempts` | Retry = continuation |
| Task-specific stop at encounter end (Sec. 5.6.2, 5.6.8) | adapted | horizon / death / separator stops | `infer_trajectories` must be task-independent; ACES handles windows |
| Eq. 2 right-censored probability | adapted | `_aggregate` | Extended to futures outside the task population; finite fallback |
| Lab values sampled uniformly within the decile (App. G) | ported | `decode` | Train min/max bound the outer deciles |
| Task outcome definitions (ICD code sets, emergent encounters, etc., Sec. 5.6) | adapted | ACES via `aces-relaxations.extract` | MEDS tasks are given as ACES criteria |
| Calibration/ECE, plausibility metrics, baselines (Sec. 5.4, 5.6.1) | omitted | — | Evaluation-side analyses; MEDS-DEV owns evaluation |

## Zero-shot materialization

- What `infer_trajectories` generates (samples per row, generation horizon, stopping rule): for every requested
  `(subject_id, prediction_time)` it builds a prompt from the prepared tokens with time ≤ prediction_time (plus
  the demographic prefix), then appends the real elapsed-gap time tokens. It samples `n_samples` = 20 futures
  at temperature 1, with no top-k/top-p and a per-key RNG seeded from `(seed, subject_id, prediction_time)`, so
  results don't depend on row order. Each future runs for up to 3 attempts × 2,000 tokens. It stops at
  MEDS_DEATH (`death`), the separator (`end_of_record`), generated time beyond prediction_time + 365 d
  (`horizon`), or the token budget (`token_budget`). Decoded events go to `events.parquet`. `samples.parquet`
  holds, per sample, the stop reason, the final clock (`end_time`), the token counts and the invalid-token count.
- How `predict_zero_shot_materialized` evaluates the task on history plus futures, and aggregates: each
  (key, sample) becomes a pseudo-subject. Its events are the real MEDS events with time ≤ prediction_time (or
  static), plus that sample's generated events. The **original** ACES task (criteria + dataset predicates) runs
  on these pseudo-subjects with `aces_relaxations.extract`, and only the row at the key's prediction time is
  kept. A future is *resolved* if it is positive, or negative with `extract_label_window_bounds`' label-window
  end ≤ its `end_time` (or it ended in death). An absent row, meaning the future violated a post-prediction
  criterion or never closed an event-bounded window, is unresolved. Probability = positives / resolved; a key
  with no resolved future gets `fallback_probability` (0.5) and is listed in `unresolved_keys.arrow`.
- Ledger per evaluated task (sampling approximation; horizon or unresolvable-future effects): `relaxations.yaml`
  has `exact: false` with four steps: Monte Carlo (n, temperature), generation limit (horizon/token budget;
  censoring), task population (excluded futures; only the label window, not criteria windows, is checked
  against the generation end) and fallback (count of unresolved keys, fraction of resolved futures). The demo
  task's realized ledger will be recorded after the demo run.
- Tasks that cannot be evaluated on generated data, and why: tasks whose post-prediction (normal-form) predicates
  match no generable code or value are rejected before any work (`_check_generable`). Examples are tasks on
  chartevents vitals (`LAB//220645//mEq/L` sodium, `LAB//227443` bicarbonate in MEDS-DEV's predicates) or on
  eMAR/infusion codes, which the paper's categories exclude. Predicates on `other_cols` are rejected too.
  MEDS-DEV's `mortality/in_icu/first_24h` needs only ICU admission/discharge and MEDS_DEATH, which are generable.

## Inputs and predicates

- MEDS assumptions: a split-sharded MEDS dataset (`data/<split>/*.parquet`) with `subject_id`, `time`, `code`,
  `numeric_value`; a `train` split is required, `tuning` is optional (early stopping). `MEDS_BIRTH` gives age;
  static `GENDER//*` codes give sex; an optional `race` column (MIMIC-IV-MEDS puts it on admission rows) gives
  race. MIMIC-IV-MEDS 0.1.0 code conventions are encoded in `MIMIC_IV_CATEGORIES`. Another dataset needs its own
  `categories` through `config_file`, and its codes must be validated.
- Required predicate names: none for preprocessing (it consumes no predicates; the dataset predicates argument was
  removed from the preprocess command). Prediction needs the task's predicates, as MEDS-DEV binds them: the
  task criteria plus `external_predicates_file` (MEDS-DEV MIMIC-IV `predicates.yaml` for the demo, which
  supplies `icu_admission`/`icu_discharge`).
- Predicate interpretation and derived features: predicates are interpreted only by ACES on real history +
  generated events, never as features. Generated lab values are sampled within their decile, so value
  thresholds are evaluated on sampled values.

## Intermediate artifacts

| Command | Payload format/version | Compatibility check | Inputs/provenance |
|---|---|---|---|
| `preprocess_data` | `curiosity-tokens/v1`: `vocabulary.json`, `tokens/<split>/<shard>.parquet` (`subject_id`, `tokens`, `times`), `payload.json` | — | `payload.json`: vocabulary digest, fitted split (`train`), SHA-256 of every MEDS shard, per-split subject/token counts |
| `pretrain` | `curiosity-qwen2/v1`: `model/` (safetensors + Qwen2 config), `payload.json` | data payload format | vocabulary digest; per-epoch train/tuning loss, optimizer steps, measured parameter count, train tokens per epoch, device |
| `infer_trajectories` | `curiosity-trajectories/v1`: `events.parquet`, `samples.parquet`, `payload.json` | data and model payload formats; model vocabulary digest == data vocabulary digest; core config check | vocabulary digest; n, sampling policy, horizon, token budget, prompt rule, time units, stop reasons |
| `predict_zero_shot_materialized` | `curiosity-predictions/v1`: `predictions.parquet`, `unresolved_keys.arrow`, `relaxations.yaml` | trajectories format and vocabulary digest; every requested key has exactly `n_samples` samples; task generability | core records task provenance and semantics digest |

## Scientific deviations

1. Dataset and vocabulary: Epic Cosmos (118M patients; ICD-10-CM/ATC/LOINC/CPT) is replaced by MIMIC-IV with
   MIMIC code systems (decisions above). No ATC, LOINC or CPT mapping is done; chief complaints and
   years-since-1970 are absent. Death is a token.
2. Scale: the paper's token budgets (90B-1.7T) are unreachable; training stops early on tuning loss.
3. Task-independent generation: the paper stops some generations at the end of an encounter for a specific task.
   Here a single horizon (365 d) and token budget serve every task, and ACES decides when the task resolves.
4. Probability: Eq. 2 is extended to futures that leave the task population, and a finite fallback (0.5)
   replaces the paper's exclusion of patients with no resolved generation.
5. Routine defaults for every hyperparameter the paper leaves unspecified (table above).

## Verification

### Primary behavior

- Signal and expected behavior: in the synthetic ICU fixture, a medication ordered one hour after ICU
  admission (`MEDICATION//START//SIGNAL_DRUG` vs `CONTROL_DRUG`) decides in-ICU death vs ICU discharge 60-90 h
  after admission. The model is pretrained without labels. Its futures must carry death after the signal drug,
  so the ACES in-ICU-mortality task scores signal subjects higher.
- How the signal enters the algorithm's selected codes/features: both drug codes match the `medication`
  category and are asserted to be vocabulary codes before the behavioral assertion; MEDS_DEATH must appear
  among generated events.
- Example generator used or adaptation made: core `build_signal_dataset` was replaced (its `BG//*`/`SIGNAL//POS`
  codes are outside the paper's categories, and a zero-shot task needs a temporal structure). The replacement
  is `tests/e2e.build_fixture`: MIMIC-IV-MEDS-style codes with ACES-derived labels, plus core split canaries.
- Positive-control result: pending the final local gate run.
- Shuffled-label negative-control result: pending the final local gate run.

| Suite/run | Exact command | Pass/skip counts | Coverage/result |
|---|---|---|---|
| Local | `uv run pytest -rs` | pending | pending |
| MEDS-DEV | `uv run pytest -m meds_dev -rs` | pending | pending |
| MIMIC-IV demo | `uv run pytest -m real_data -rs` | not run: awaiting human approval | — |
| Full target dataset (MIMIC-IV; revision to be confirmed) | not run | — | Blocked: no evaluation task bundle was supplied |

## Review gates

| Gate | Evidence reviewed | Reviewer/approval | Date or reference |
|---|---|---|---|
| Before MIMIC-IV demo | Source fidelity, local/MEDS-DEV results, predicates, demo plan | Requested from Florent; not yet approved | 2026-11-09 |
| Before full target dataset | Demo result, target-dataset/task/predicate compatibility, access, full-run resources | Not reached | — |

## Resource measurements

| Environment/command | Wall seconds | Peak tree PSS bytes | Peak tree RSS bytes | GPU bytes |
|---|---:|---:|---:|---:|
| Synthetic fixture (pending final run) | pending | pending | pending | n/a (CPU) |

## Environment and provenance

- Algorithm commit and dirty state: recorded per artifact manifest; branch `worktree-curiosity-zero-shot`.
- Dataset/task identifiers: synthetic fixture only; MIMIC-IV demo pending approval.
- Dataset, task, and predicate revisions: MEDS-DEV pinned by core (`19a8aa34`) for MEDS-DEV and demo tests.
- Task-bundle kind (clinical ACES tasks / sampled code-by-horizon tasks) and the commit of the tool that built it:
  none supplied (evaluation deferred).
- Shared task-bundle identifier/path: none supplied.
- Digest of the bundle's task definitions: none supplied.
- Cluster checkout commit and job IDs: none (no cluster run).
- Structured-result branch/PR: none.
- Dependency/platform/hardware summary: Python 3.12, torch 2.15.0+cu132, transformers 5.22.2, polars 1.44.2,
  es-aces 0.7.4; aarch64 NVIDIA GB10 host (tests run on CPU with one torch thread).

## Known limitations

- Not a reproduction of the paper's results: different data, vocabulary and scale.
- Only the label window, not criteria windows, is checked against the generation end when resolving a negative.
- Generation runs one key at a time (its n samples are batched); full-data inference cost is
  O(keys × n × tokens). Prediction duplicates each key's real history n times, in chunks of 256 keys.
- Generated diagnosis sub-token sequences can form ICD codes that do not exist; they are emitted as generated
  (invalid-token counts are recorded per sample).
