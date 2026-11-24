"""Curiosity configuration. Paper = arXiv 2611.15710 (Curiosity-0826-4B), which states that every representation
it does not describe is unchanged from Curiosity-0226-1B (arXiv 2508.12104v3). "Routine" marks defaults the papers
leave unspecified (IMPLEMENTATION_REPORT.md)."""

from dataclasses import dataclass

from meds_algorithm import NNConfig

# The paper's event categories, in matching order. Which MEDS codes belong to each is dataset knowledge: the ACES
# plain predicate ``curiosity_<category>`` in a dataset category file (``datasets/<dataset_name>.yaml`` or
# ``additional_predicates_file``); ``null`` marks a category the dataset does not record. An event takes the first
# matching category. Vital categories carry their source unit; values are converted to the paper's units
# (Table 2: mmHg, inches, pounds, kg/m2).
CATEGORIES = (
    "sex",
    "birth",
    "gestational_age",  # numeric, completed weeks
    "birth_weight",  # numeric
    "death",
    "cause_of_death",
    "encounter",
    "diagnosis",
    "lab",
    "vital_sbp_mmhg",
    "vital_dbp_mmhg",
    "vital_height_in",
    "vital_height_cm",
    "vital_weight_lb",
    "vital_weight_kg",
    "vital_bmi",
    "medication",
    "reported_medication",
    "procedure",
)

# Paper inclusion thresholds as absolute counts over Epic Cosmos (Sec. 2.3-2.6), divided by the matching category
# total of Table 3, so that a dataset of any size keeps the same share of its events.
_LABS_NUMERIC, _LABS_TEXT, _MEDICATIONS, _PROCEDURES = 31.5e9, 5.96e9, 9.43e9, 16.5e9


@dataclass(frozen=True, kw_only=True)
class CuriosityConfig(NNConfig):
    """Curiosity settings.

    Capacity mapping (reference = Curiosity-S of the 0226 paper, Table 3: 6 layers, width 768, 12 heads, MLP 3072;
    the 0826 paper gives only 40 layers and ~4.3B parameters): ``heads = max(1, round(reference_heads * capacity))``,
    ``hidden = heads * head_dim``, ``intermediate = mlp_ratio * hidden``; depth stays ``num_layers``.
    ``batch_size`` is sequences per device microbatch; ``sequences_per_step`` is reached by gradient accumulation.
    ``dropout`` is Qwen2's attention dropout (Qwen2 has no other dropout).
    """

    # NNConfig: routine defaults (papers: learning rate and batch size were swept, values not given).
    learning_rate: float = 6e-4
    batch_size: int = 8
    weight_decay: float = 0.1

    # Tokenization thresholds (paper Sec. 2.3-2.6), as shares of the category's training events.
    lab_numeric_min_share: float = 400_000 / _LABS_NUMERIC  # numeric labs with >= 400,000 events
    lab_text_min_share: float = 22_000 / _LABS_TEXT  # non-numeric labs with >= 22,000 events
    lab_text_value_min_share: float = 10_000 / _LABS_TEXT  # rarer result values -> the Other token
    lab_min_bucket_observations: int = 500  # absolute: lab-unit pairs with fewer values are excluded
    procedure_min_share: float = 10_000 / _PROCEDURES
    medication_attribute_min_share: float = 10_000 / _MEDICATIONS  # rarer route/dose/frequency -> null
    race_column: str | None = "race"
    # Right-censoring (paper Sec. 2.2.2): a record without death is padded with six-month time tokens up to the
    # data cutoff. Cutoff per subject = min(latest training time, last event + follow_up_days); MIMIC-IV records
    # out-of-hospital deaths for one year after the last discharge.
    follow_up_days: float = 365.0

    # Architecture (Qwen2, random initialization).
    num_layers: int = 6
    reference_heads: int = 12
    head_dim: int = 64
    mlp_ratio: int = 4
    context_length: int = 8192  # stage 1 (paper Sec. 3)
    # Stage 2 (paper Sec. 3.2): unscaled RoPE, context extended for extension_steps optimizer steps plus
    # extension_epochs epochs. 0 steps and 0 epochs disable it.
    extended_context_length: int = 16384
    extension_steps: int = 1000
    extension_epochs: float = 1.5

    # Pretraining (stage 1): batch 512 sequences, cosine schedule with 10x decay (0226 paper); rest routine.
    sequences_per_step: int = 512
    max_epochs: int = 20
    early_stopping_patience: int = 2
    warmup_fraction: float = 0.01
    final_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    device: str = "auto"

    # Generation (paper Sec. 4.1: 60 simulations, up to 4,000 tokens, four retries for 365-day mortality).
    n_samples: int = 60
    temperature: float = 1.0
    max_new_tokens: int = 4000
    max_attempts: int = 5
    horizon_days: float = 365.0

    # Prediction: probability for a key none of whose futures resolves the task (flagged in unresolved_keys.arrow).
    fallback_probability: float = 0.5

    @property
    def num_heads(self) -> int:
        return max(1, round(self.reference_heads * self.capacity))

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def generation_context(self) -> int:
        """Context available at inference: the extended length when stage 2 runs."""
        return self.extended_context_length if self.extension_steps or self.extension_epochs else self.context_length

    def validate(self):
        super().validate()
        positive = (
            "num_layers", "reference_heads", "head_dim", "mlp_ratio", "sequences_per_step", "max_epochs", "n_samples",
            "max_new_tokens", "max_attempts", "lab_min_bucket_observations",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        shares = ("lab_numeric_min_share", "lab_text_min_share", "lab_text_value_min_share", "procedure_min_share",
                  "medication_attribute_min_share")
        for name in shares:
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.sequences_per_step % self.batch_size:
            raise ValueError("sequences_per_step must be a multiple of batch_size")
        if self.extended_context_length < self.context_length:
            raise ValueError("extended_context_length must be at least context_length")
        if self.extension_steps < 0 or self.extension_epochs < 0:
            raise ValueError("extension_steps and extension_epochs must be nonnegative")
        if self.generation_context <= self.max_new_tokens:
            raise ValueError("the generation context must exceed max_new_tokens to leave room for the prompt")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be nonnegative")
        if not 0 <= self.warmup_fraction < 1 or not 0 < self.final_lr_ratio <= 1:
            raise ValueError("warmup_fraction must be in [0, 1) and final_lr_ratio in (0, 1]")
        if self.gradient_clip <= 0 or self.temperature <= 0 or self.horizon_days <= 0 or self.follow_up_days < 0:
            raise ValueError("gradient_clip, temperature and horizon_days must be positive, follow_up_days nonnegative")
        if not 0 <= self.fallback_probability <= 1:
            raise ValueError("fallback_probability must be in [0, 1]")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
