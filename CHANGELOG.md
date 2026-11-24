# Changes

## 0.1.1 (2026-11-24)

- An `other` category (on by default; `other_category=False` restores the papers' vocabulary): every code no paper
  category matches, e.g. ICU charting, is kept as a whole-code token with a shared value bucket, latest value per
  code in each 6-hour window. Tasks on such codes (mean arterial pressure for hypotension) can be answered.
- A dataset needs no category file: without one, every code is `other` and death is the MEDS standard `MEDS_DEATH`.
  A category file only adds the papers' handling (ICD/ATC splits, lab and vital tokens, demographics).

## 0.1.0 (2026-11-24)

- Curiosity-0826 (arXiv 2611.15710) as a `zero_shot_materialized` MEDS algorithm, replacing the earlier
  Curiosity-0226 port: BOS-first sequences, age/year at start of timeline with pediatric buckets, birth,
  gestational age and birth weight, death with causes and EOS, right-censoring padding, lab collection markers,
  30-bucket labs, non-numeric labs, binned vitals, ATC-split medications with attributes and a patient-reported
  flag, and two-stage context extension to 16,384 tokens with unscaled RoPE.
- Prompts are built from what is recorded at or before each prediction time; prompt-level temporal-invariance
  tests complement the shared future-blind conformance rerun.
- Model size presets (`size`: small, medium, large, xlarge): Curiosity-0226 S/M/L and a 40-layer ~4.2B model for
  Curiosity-0826-4B.
- MIMIC-IV category file matches MIMIC-IV-MEDS 0.4.0 (MEDS-DEV's current build) and 0.1.0; race also comes from
  `RACE//` events.
- `aces-relaxations` from PyPI (0.1.0); `meds-algorithm` with the temporal-leakage conformance (0.4.0a6).
