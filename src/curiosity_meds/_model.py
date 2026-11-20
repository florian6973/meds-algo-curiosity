"""Qwen2 decoder (paper Sec. 5.3.1): construction, packed next-token pretraining and Monte Carlo generation."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .config import CuriosityConfig

IGNORE = -100


def _device(config: CuriosityConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _seeded(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config: CuriosityConfig, vocab_size: int, pad_id: int):
    """Randomly initialized Qwen2ForCausalLM. Embeddings are tied: this reproduces the paper's 62M/119M counts."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    hidden = config.hidden_size
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=hidden,
            intermediate_size=config.mlp_ratio * hidden,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            max_position_embeddings=config.context_length,
            attention_dropout=config.dropout,
            tie_word_embeddings=True,
            pad_token_id=pad_id,
            use_cache=True,
        )
    )


def load_model(directory: Path, config: CuriosityConfig):
    from transformers import Qwen2ForCausalLM

    return Qwen2ForCausalLM.from_pretrained(directory).to(_device(config)).eval()


def _pack(sequences: list[np.ndarray], sep_id: int, pad_id: int, length: int) -> np.ndarray:
    """Dense packing (paper Sec. 5.3.1): subjects joined by the separator token, cut into ``length`` windows."""
    if not sequences:
        return np.empty((0, length), dtype=np.int64)
    stream = np.concatenate([np.append(sequence, sep_id) for sequence in sequences]).astype(np.int64)
    windows = math.ceil(len(stream) / length)
    padded = np.full(windows * length, pad_id, dtype=np.int64)
    padded[: len(stream)] = stream
    return padded.reshape(windows, length)


def _loss(model, batch: torch.Tensor, pad_id: int, device: torch.device) -> torch.Tensor:
    batch = batch.to(device)
    labels = batch.masked_fill(batch == pad_id, IGNORE)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(input_ids=batch, labels=labels).loss


@torch.no_grad()
def _evaluate(model, windows: np.ndarray, config: CuriosityConfig, pad_id: int, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(windows), config.batch_size):
        batch = torch.from_numpy(windows[start : start + config.batch_size])
        tokens = int((batch[:, 1:] != pad_id).sum())
        total += float(_loss(model, batch, pad_id, device)) * tokens
        count += tokens
    return total / max(count, 1)


def pretrain(train: list[np.ndarray], tuning: list[np.ndarray], config: CuriosityConfig, vocab_size: int,
             sep_id: int, pad_id: int, seed: int):
    """AdamW with linear warmup then cosine decay to ``final_lr_ratio`` (paper: 10x decay); keep the epoch with
    the lowest tuning loss and stop after ``early_stopping_patience`` epochs without improvement."""
    _seeded(seed)
    device = _device(config)
    model = build_model(config, vocab_size, pad_id).to(device)
    rng = np.random.default_rng(seed)
    tuning_windows = _pack(tuning, sep_id, pad_id, config.context_length)
    windows_per_epoch = len(_pack(train, sep_id, pad_id, config.context_length))
    steps_per_epoch = max(1, math.ceil(windows_per_epoch / config.sequences_per_step))
    total_steps = steps_per_epoch * config.max_epochs
    warmup = int(config.warmup_fraction * total_steps)

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return config.final_lr_ratio + (1 - config.final_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history, best, best_state, stale, step = [], math.inf, None, 0, 0
    for epoch in range(config.max_epochs):
        model.train()
        order = rng.permutation(len(train))
        windows = _pack([train[i] for i in order], sep_id, pad_id, config.context_length)
        train_loss = []
        for start in range(0, len(windows), config.sequences_per_step):
            optimizer.zero_grad(set_to_none=True)
            chunk = windows[start : start + config.sequences_per_step]
            for micro in range(0, len(chunk), config.batch_size):
                loss = _loss(model, torch.from_numpy(chunk[micro : micro + config.batch_size]), pad_id, device)
                (loss * len(chunk[micro : micro + config.batch_size]) / len(chunk)).backward()
                train_loss.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            step += 1
        tuning_loss = _evaluate(model, tuning_windows, config, pad_id, device) if len(tuning_windows) else None
        history.append({"epoch": epoch, "step": step, "train_loss": float(np.mean(train_loss)), "tuning_loss": tuning_loss})
        if tuning_loss is None:
            continue
        if tuning_loss < best:
            best, best_state, stale = tuning_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "history": history,
        "optimizer_steps": step,
        "best_tuning_loss": None if best_state is None else best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_tokens_per_epoch": int(sum(len(s) + 1 for s in train)),
        "device": str(device),
    }
    return model.cpu(), summary


def key_seed(seed: int, subject_id: int, prediction_time_us: int) -> int:
    """Per-key RNG seed, independent of row order and batch composition."""
    digest = hashlib.sha256(f"{seed}:{subject_id}:{prediction_time_us}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


@torch.no_grad()
def generate(model, prompt: list[int], config: CuriosityConfig, seed: int, token_seconds: np.ndarray,
             stop_ids: dict[int, str]) -> list[tuple[list[int], str]]:
    """``n_samples`` futures continuing ``prompt`` at ``temperature`` (no top-k/top-p, paper Sec. 5.5).

    Up to ``max_attempts`` attempts of ``max_new_tokens`` each; every attempt re-prompts with the most recent
    ``context_length - max_new_tokens`` tokens (left truncation). A sample stops at a stop token (death, end of
    record), once generated time passes the horizon, or when the token budget is spent.
    """
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    n = config.n_samples
    seconds = torch.as_tensor(token_seconds, dtype=torch.float64, device=device)
    horizon = config.horizon_days * 86400.0
    sequence = torch.tensor(prompt, dtype=torch.long, device=device).repeat(n, 1)
    elapsed = torch.zeros(n, dtype=torch.float64, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    reasons = ["token_budget"] * n
    start = sequence.shape[1]
    for _ in range(config.max_attempts):
        context = sequence[:, -(config.context_length - config.max_new_tokens):]
        output = model(input_ids=context, use_cache=True)
        for _ in range(config.max_new_tokens):
            probabilities = torch.softmax(output.logits[:, -1].float() / config.temperature, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator)
            sequence = torch.cat([sequence, token], dim=1)
            lengths += active
            elapsed += torch.where(active, seconds[token[:, 0]], 0.0)
            for index in torch.nonzero(active).flatten().tolist():
                reason = stop_ids.get(int(token[index, 0]))
                if reason is None and float(elapsed[index]) > horizon:
                    reason = "horizon"
                if reason is not None:
                    reasons[index] = reason
                    active[index] = False
            if not active.any():
                break
            output = model(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
        if not active.any():
            break
    return [(sequence[i, start : start + int(lengths[i])].tolist(), reasons[i]) for i in range(n)]
