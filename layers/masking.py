""" hybrid phase masking and learned mask-token replacement."""

from __future__ import annotations

import torch
import torch.nn as nn


def _target_counts(batch_size, period_len, ratio, device):
    ratios = torch.as_tensor(ratio, dtype=torch.float32, device=device)
    if ratios.ndim == 0:
        ratios = ratios.expand(int(batch_size))
    if ratios.shape != (int(batch_size),):
        raise ValueError("ratio must be scalar or have one value per sample")
    if bool(((ratios < 0) | (ratios > 1)).any()):
        raise ValueError("ratio must be in [0, 1]")
    return torch.round(ratios * int(period_len)).long()


def point_mask(batch_size, period_len=96, ratio=0.30, device=None, generator=None):
    ratios = torch.as_tensor(ratio, dtype=torch.float32, device=device)
    if ratios.ndim == 0:
        ratios = ratios.expand(int(batch_size))
    if ratios.shape != (int(batch_size),):
        raise ValueError("ratio must be scalar or have one value per sample")
    if bool(((ratios < 0) | (ratios > 1)).any()):
        raise ValueError("ratio must be in [0, 1]")
    return torch.rand(
        int(batch_size), int(period_len), device=device, generator=generator
    ) < ratios.unsqueeze(1)


def circular_block_mask(
    batch_size, period_len=96, ratio=0.50, min_span=8, max_span=24,
    device=None, generator=None,
):
    """Sample circular blocks until each row reaches its target coverage."""
    batch_size, period_len = int(batch_size), int(period_len)
    min_span, max_span = int(min_span), int(max_span)
    if period_len <= 0 or batch_size < 0:
        raise ValueError("period_len must be positive and batch_size non-negative")
    if not 0 < min_span <= max_span <= period_len:
        raise ValueError("require 0 < min_span <= max_span <= period_len")
    target = _target_counts(batch_size, period_len, ratio, device)
    mask = torch.zeros(batch_size, period_len, dtype=torch.bool, device=device)
    offsets = torch.arange(max_span, device=device).view(1, -1)
    while bool((mask.sum(dim=1) < target).any()):
        active_rows = (mask.sum(dim=1) < target).nonzero().flatten()
        starts = torch.randint(
            0, period_len, (active_rows.numel(), 1), device=device, generator=generator
        )
        spans = torch.randint(
            min_span, max_span + 1, (active_rows.numel(), 1),
            device=device, generator=generator,
        )
        indices = (starts + offsets).remainder(period_len)
        proposed = (offsets < spans) & ~mask[active_rows.unsqueeze(1), indices]
        remaining = target[active_rows] - mask[active_rows].sum(dim=1)
        accepted = proposed & (proposed.cumsum(dim=1) <= remaining.unsqueeze(1))
        mask[active_rows.unsqueeze(1), indices] |= accepted
    return mask


def single_block_mask(batch_size, period_len, ratio, device=None, generator=None):
    target = _target_counts(batch_size, period_len, ratio, device)
    starts = torch.randint(
        0, int(period_len), (int(batch_size), 1), device=device, generator=generator
    )
    offsets = torch.arange(int(period_len), device=device).view(1, -1)
    indices = (starts + offsets).remainder(int(period_len))
    selected = offsets < target.unsqueeze(1)
    mask = torch.zeros(int(batch_size), int(period_len), dtype=torch.bool, device=device)
    mask.scatter_(1, indices, selected)
    return mask


def mixed_mask(
    batch_size, period_len, ratio, block_fraction, min_span, max_span,
    device=None, generator=None,
):
    target = _target_counts(batch_size, period_len, ratio, device)
    block_target = torch.round(target.float() * float(block_fraction)).long()
    block_ratio = block_target.float() / int(period_len)
    mask = circular_block_mask(
        batch_size, period_len, block_ratio, min_span, max_span, device, generator
    )
    remaining = target - mask.sum(dim=1)
    scores = torch.rand(int(batch_size), int(period_len), device=device, generator=generator)
    scores = scores.masked_fill(mask, -1.0)
    order = scores.argsort(dim=1, descending=True)
    ranks = torch.arange(int(period_len), device=device).view(1, -1)
    selected = ranks < remaining.unsqueeze(1)
    point_part = torch.zeros_like(mask)
    point_part.scatter_(1, order, selected)
    return mask | point_part


class GlucoScopeMaskGenerator:
    """Dynamic per-sample masking with one sampled strategy per sample."""

    STRATEGY_NAMES = ("point", "multi_block", "single_block", "mixed")

    def __init__(
        self, ratio_min=0.0, ratio_max=0.5,
        point_probability=0.2, multi_block_probability=0.3,
        single_block_probability=0.2, mixed_probability=0.3,
        min_span=8, max_span=24, mixed_block_fraction=0.5,
    ):
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.probabilities = torch.tensor(
            [point_probability, multi_block_probability,
             single_block_probability, mixed_probability], dtype=torch.float32,
        )
        if not 0 <= self.ratio_min <= self.ratio_max <= 1:
            raise ValueError("require 0 <= ratio_min <= ratio_max <= 1")
        if bool((self.probabilities < 0).any()) or not torch.isclose(
            self.probabilities.sum(), torch.tensor(1.0)
        ):
            raise ValueError("mask strategy probabilities must be nonnegative and sum to 1")
        self.min_span = int(min_span)
        self.max_span = int(max_span)
        self.mixed_block_fraction = float(mixed_block_fraction)
        if not 0 <= self.mixed_block_fraction <= 1:
            raise ValueError("mixed_block_fraction must be in [0, 1]")

    def __call__(
        self, batch_size, period_len, device, effective_days=None, generator=None,
    ):
        batch_size = int(batch_size)
        ratios = self.ratio_min + (self.ratio_max - self.ratio_min) * torch.rand(
            batch_size, device=device, generator=generator
        )
        strategies = torch.multinomial(
            self.probabilities.to(device), batch_size, replacement=True,
            generator=generator,
        )
        candidates = (
            point_mask(batch_size, period_len, ratios, device, generator),
            circular_block_mask(
                batch_size, period_len, ratios, self.min_span, self.max_span,
                device, generator,
            ),
            single_block_mask(batch_size, period_len, ratios, device, generator),
            mixed_mask(
                batch_size, period_len, ratios, self.mixed_block_fraction,
                self.min_span, self.max_span, device, generator,
            ),
        )
        mask = candidates[0]
        for index, candidate in enumerate(candidates[1:], start=1):
            mask = torch.where((strategies == index).unsqueeze(1), candidate, mask)
        return mask, strategies


class GlucoScopeMasker(nn.Module):
    """Replace every selected token; V1 has no 80/10/10 shortcut."""

    def __init__(self, token_dim=64):
        super().__init__()
        self.token_dim = int(token_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, tokens, phase_mask=None):
        if phase_mask is None:
            return tokens
        if phase_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"Expected mask {tuple(tokens.shape[:2])}, got {tuple(phase_mask.shape)}"
            )
        return torch.where(
            phase_mask.bool().unsqueeze(-1), self.mask_token.to(tokens), tokens
        )
