"""Curiosity configuration. Paper = arXiv 2611.15710 (Curiosity-0826-4B), which states that every representation
it does not describe is unchanged from Curiosity-0226-1B (arXiv 2508.12104v3). "Routine" marks defaults the papers
leave unspecified (IMPLEMENTATION_REPORT.md)."""

from dataclasses import dataclass
from pathlib import Path

from meds_algorithm import NNConfig

# Model sizes: (layers, attention heads of 64 dimensions; MLP = 4 x width). small, medium and large are
# Curiosity-0226 S/M/L (2508.12104v3 Table 3: 62M, 119M, 1B parameters at its 7,105-token vocabulary). xlarge is
# Curiosity-0826-4B: 40 layers and ~4.3B parameters (2611.15710 Sec. 3.1); its width is not reported, and 2,560
# (40 heads) is the width that gives ~4.2B parameters at 40 layers.
SIZES = {"small": (6, 12), "medium": (12, 12), "large": (16, 32), "xlarge": (40, 40)}


@dataclass(frozen=True, kw_only=True)
class CuriosityConfig(NNConfig):
    """Curiosity settings.

    ``size`` preselects the architecture (``SIZES``: small, medium, large, xlarge); ``num_layers`` and
    ``reference_heads`` override its depth and heads. Then ``heads = max(1, round(reference_heads * capacity))``,
    ``hidden = heads * head_dim``, ``intermediate = mlp_ratio * hidden``; depth is not scaled by ``capacity``.
    ``batch_size`` is sequences per device microbatch; ``sequences_per_step`` is reached by gradient accumulation.
    ``dropout`` is Qwen2's attention dropout (Qwen2 has no other dropout).
    """

    # NNConfig: routine defaults (papers: learning rate and batch size were swept, values not given).
    learning_rate: float = 6e-4
    batch_size: int = 8
    weight_decay: float = 0.1

    # Tokenization (dataset-agnostic adaptation, IMPLEMENTATION_REPORT.md). A code, a text result value or an
    # attribute value seen fewer than max(min_code_count, min_code_share x training events) times is rare: a rare code
    # becomes [UNK] (as in the paper), a rare result RESULT//[OTHER], a rare attribute value no token. A code gets the
    # 30 value buckets when it has at least min_bucket_observations numeric values (0826 Sec. 2.3.3: 500). The latest
    # value of each numeric code in each downsample_hours window is kept (0826 Sec. 2.4 does this for vitals; None
    # keeps every value).
    min_code_count: int = 10
    min_code_share: float = 1e-6
    min_bucket_observations: int = 500
    downsample_hours: float | None = 6.0
    # Optional YAML of hierarchy rules: codes a rule matches are split after the rule's matched prefix, on a separator
    # or into character widths, into a head token and depth-tagged sub-tokens (rare parts truncate the code to its
    # known ancestor). None keeps every code whole. Example: curiosity_meds/hierarchies/MIMIC-IV.yaml (ICD 3/2/2).
    hierarchy_file: Path | None = None
    race_column: str | None = "race"
    # Right-censoring (paper Sec. 2.2.2): a record without death is padded with six-month time tokens up to the
    # data cutoff. Cutoff per subject = min(latest training time, last event + follow_up_days); MIMIC-IV records
    # out-of-hospital deaths for one year after the last discharge.
    follow_up_days: float = 365.0

    # Architecture (Qwen2, random initialization).
    size: str = "small"
    num_layers: int | None = None  # None: the size's depth
    reference_heads: int | None = None  # None: the size's heads
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

    # Generation: 100 futures (chosen by Florent; the paper uses 20-80 per task, 60 for 365-day mortality, Sec. 4.1),
    # up to 4,000 tokens and four retries (the paper's 365-day mortality setting).
    n_samples: int = 100
    temperature: float = 1.0
    max_new_tokens: int = 4000
    max_attempts: int = 5
    horizon_days: float = 365.0

    # Prediction: probability for a key none of whose futures resolves the task (flagged in unresolved_keys.arrow).
    fallback_probability: float = 0.5

    @property
    def layers(self) -> int:
        return self.num_layers if self.num_layers is not None else SIZES[self.size][0]

    @property
    def num_heads(self) -> int:
        heads = self.reference_heads if self.reference_heads is not None else SIZES[self.size][1]
        return max(1, round(heads * self.capacity))

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def generation_context(self) -> int:
        """Context available at inference: the extended length when stage 2 runs."""
        return self.extended_context_length if self.extension_steps or self.extension_epochs else self.context_length

    def validate(self):
        super().validate()
        if self.size not in SIZES:
            raise ValueError(f"size must be one of {sorted(SIZES)}")
        for name in ("num_layers", "reference_heads"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        positive = (
            "head_dim", "mlp_ratio", "sequences_per_step", "max_epochs", "n_samples",
            "max_new_tokens", "max_attempts", "min_bucket_observations", "min_code_count",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.min_code_share < 1:
            raise ValueError("min_code_share must be in [0, 1)")
        if self.sequences_per_step % self.batch_size:
            raise ValueError("sequences_per_step must be a multiple of batch_size")
        if self.extended_context_length < self.context_length:
            raise ValueError("extended_context_length must be at least context_length")
        if self.extension_steps < 0 or self.extension_epochs < 0:
            raise ValueError("extension_steps and extension_epochs must be nonnegative")
        if self.generation_context <= self.max_new_tokens:
            raise ValueError("the generation context must exceed max_new_tokens to leave room for the prompt")
        if self.downsample_hours is not None and self.downsample_hours <= 0:
            raise ValueError("downsample_hours must be positive or None")
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
