"""Independent Multicenter classifier built on pretrained global features."""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.glucoscope import GlucoScope


def _cfg(config, key, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class GlucoScopeGlobalFeatureClassifier(nn.Module):
    """Three-class MLP consuming the backbone's pretrained global feature."""

    MODEL_NAME = "GlucoScopeGlobalFeatureClassifier"

    def __init__(self, config, backbone=None):
        super().__init__()
        self.backbone = backbone or GlucoScope(config)
        latent = int(self.backbone.latent_dim)
        hidden = int(_cfg(config, "head_hidden", 128))
        dropout = float(_cfg(config, "head_dropout", 0.1))
        classes = int(_cfg(config, "num_classes", 3))
        if classes != 3:
            raise ValueError("Global-feature experiment requires num_classes=3")
        self.head = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def _encode_single(self, x):
        phase_states, global_feature = self.backbone.encode(x)
        _, attention = self.backbone.global_pooling(phase_states)
        return global_feature, attention, phase_states

    def encode_for_head(self, x):
        if x.dim() == 3:
            return self._encode_single(x)
        if x.dim() != 4:
            raise ValueError("Expected (B,96,7) or (B,V,96,7)")
        batch, views, phases, features = x.shape
        global_feature, attention, phase_states = self._encode_single(
            x.reshape(batch * views, phases, features)
        )
        return (
            global_feature.reshape(batch, views, -1).mean(dim=1),
            attention.reshape(batch, views, phases).mean(dim=1),
            phase_states.reshape(batch, views, phases, -1).mean(dim=1),
        )

    def forward(self, x, return_features=False):
        global_feature, attention, phase_states = self.encode_for_head(x)
        logits = self.head(global_feature)
        if return_features:
            return {
                "prediction": logits,
                "pooled_representation": global_feature,
                "global_feature": global_feature,
                "phase_representation": phase_states,
                "attention_weights": attention,
            }
        return logits
