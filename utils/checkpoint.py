"""GlucoScope checkpoint creation and strict loading."""

from __future__ import annotations

from pathlib import Path

import torch


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def load_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    return checkpoint, state


def load_backbone(model, path, strict=True):
    checkpoint, state = load_checkpoint(path)
    model.load_state_dict(state, strict=strict)
    return checkpoint
