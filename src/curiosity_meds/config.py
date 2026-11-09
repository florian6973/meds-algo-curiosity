"""Curiosity configuration. Paper = arXiv 2508.12104v3 (Waxler et al., "Generative Medical Event Models
Improve with Scale"); "routine" marks defaults the paper leaves unspecified (IMPLEMENTATION_REPORT.md)."""

import re
from dataclasses import dataclass

from meds_algorithm import NNConfig


@dataclass(frozen=True)
class CodeCategory:
    """One event family the tokenizer keeps (paper Sec. 5.2). Events are assigned to the first matching category.

    ``pattern`` is a regular expression over MEDS codes. ``top_k`` keeps only the most frequent training-split
    codes (paper: top 1,000 labs, top 1,500 procedures); other matching codes become the unknown token.
    ``numeric`` keeps only events with a numeric value and appends a shared decile token (paper Sec. 5.2.4).
    ``split_icd`` splits ``<prefix>//<ICD code>`` into 3/2/2-character sub-tokens (paper Sec. 5.2.3).
    """

    name: str
    pattern: str
    top_k: int | None = None
    numeric: bool = False
    split_icd: bool = False

    def validate(self):
        re.compile(self.pattern)
        if not self.name or (self.top_k is not None and self.top_k < 1):
            raise ValueError(f"Invalid code category {self!r}")


# Paper event families mapped onto MIMIC-IV-MEDS 0.1.0 codes (the MEDS-DEV demo and full-data ETL). Excluded,
# as in the paper (Sec. 5.2.8): chartevents (LAB//<itemid>), eMAR administrations, infusions, outputs, OMR,
# DRG, specimen collection, medication stops. Other datasets supply their own mapping through config_file.
MIMIC_IV_CATEGORIES = (
    CodeCategory("sex", r"^GENDER//"),
    CodeCategory("death", r"^MEDS_DEATH$"),
    CodeCategory(
        "encounter",
        r"^(HOSPITAL_ADMISSION//|HOSPITAL_DISCHARGE//|ICU_ADMISSION//|ICU_DISCHARGE//|TRANSFER_TO//|ED_REGISTRATION$|ED_OUT$)",
    ),
    CodeCategory("diagnosis", r"^DIAGNOSIS//ICD//(9|10)//", split_icd=True),
    CodeCategory("lab", r"^LAB//RESULT//", top_k=1000, numeric=True),
    CodeCategory("medication", r"^MEDICATION//START//"),
    CodeCategory("procedure", r"^(PROCEDURE//ICD//|HCPCS//|PROCEDURE//START//)", top_k=1500),
)


@dataclass(frozen=True, kw_only=True)
class CuriosityConfig(NNConfig):
    """Curiosity settings.

    Capacity mapping (reference = Curiosity-S, paper Table 3: 6 layers, width 768, 12 heads, MLP 3072):
    ``heads = max(1, round(reference_heads * capacity))``, ``hidden = heads * head_dim``,
    ``intermediate = mlp_ratio * hidden``; depth stays ``num_layers``. Capacity 1.0 gives 768/12/3072.
    ``batch_size`` is sequences per device microbatch; ``sequences_per_step`` (paper: 512) is reached by
    gradient accumulation. ``dropout`` is Qwen2's attention dropout (Qwen2 has no other dropout).
    """

    # NNConfig: routine defaults (paper: AdamW, "standard hyperparameters"); dropout/capacity inherited.
    learning_rate: float = 6e-4
    batch_size: int = 8
    weight_decay: float = 0.1

    # Tokenization (paper Sec. 5.2, Table 2).
    categories: tuple[CodeCategory, ...] = MIMIC_IV_CATEGORIES
    num_quantiles: int = 10
    age_bucket_years: int = 5
    race_column: str | None = "race"

    # Architecture (paper Sec. 5.3.1, Table 3; Qwen2, random initialization).
    num_layers: int = 6
    reference_heads: int = 12
    head_dim: int = 64
    mlp_ratio: int = 4
    context_length: int = 8192

    # Pretraining (paper: batch 512 sequences, cosine schedule with 10x decay; rest routine).
    sequences_per_step: int = 512
    max_epochs: int = 20
    early_stopping_patience: int = 2
    warmup_fraction: float = 0.01
    final_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    device: str = "auto"

    # Generation (paper Sec. 5.5-5.6: temperature 1, 2,000 tokens, up to two retries, n=20).
    n_samples: int = 20
    temperature: float = 1.0
    max_new_tokens: int = 2000
    max_attempts: int = 3
    horizon_days: float = 365.0

    # Prediction: probability for a key none of whose futures resolves the task (flagged in unresolved_keys.arrow).
    fallback_probability: float = 0.5

    @property
    def num_heads(self) -> int:
        return max(1, round(self.reference_heads * self.capacity))

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim

    def validate(self):
        super().validate()
        for category in self.categories:
            category.validate()
        if len({category.name for category in self.categories}) != len(self.categories):
            raise ValueError("Category names must be unique")
        positive = (
            "num_quantiles", "age_bucket_years", "num_layers", "reference_heads", "head_dim", "mlp_ratio",
            "sequences_per_step", "max_epochs", "n_samples", "max_new_tokens", "max_attempts",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_quantiles < 2:
            raise ValueError("num_quantiles must be at least 2")
        if self.sequences_per_step % self.batch_size:
            raise ValueError("sequences_per_step must be a multiple of batch_size")
        if self.context_length <= self.max_new_tokens:
            raise ValueError("context_length must exceed max_new_tokens to leave room for the prompt")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be nonnegative")
        if not 0 <= self.warmup_fraction < 1 or not 0 < self.final_lr_ratio <= 1:
            raise ValueError("warmup_fraction must be in [0, 1) and final_lr_ratio in (0, 1]")
        if self.gradient_clip <= 0 or self.temperature <= 0 or self.horizon_days <= 0:
            raise ValueError("gradient_clip, temperature and horizon_days must be positive")
        if not 0 <= self.fallback_probability <= 1:
            raise ValueError("fallback_probability must be in [0, 1]")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
