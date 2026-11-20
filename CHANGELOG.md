# Changes

## 0.1.0 (unreleased)

- Curiosity-0826 (arXiv 2611.15710) as a `zero_shot_materialized` MEDS algorithm, replacing the earlier
  Curiosity-0226 port: BOS-first sequences, age/year at start of timeline with pediatric buckets, birth,
  gestational age and birth weight, death with causes and EOS, right-censoring padding, lab collection markers,
  30-bucket labs, non-numeric labs, binned vitals, ATC-split medications with attributes and a patient-reported
  flag, and two-stage context extension to 16,384 tokens with unscaled RoPE.
- Prompts are built from what is recorded at or before each prediction time; prompt-level temporal-invariance
  tests complement the shared future-blind conformance rerun.
- `aces-relaxations` from PyPI (0.1.0); `meds-algorithm` pinned to the commit with temporal-leakage conformance.
