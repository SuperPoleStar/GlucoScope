"""All GlucoScope model structures and tensor paths."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import GlucoScopeRoutingLayer
from layers.masking import GlucoScopeMasker


def _cfg(config, key, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def build_glucoscope_fourier_encoding(period_len, token_dim):
    """Fixed integer-harmonic table ``(1,L,D)`` with exact daily period."""
    period_len, token_dim = int(period_len), int(token_dim)
    if period_len <= 0 or token_dim <= 0 or token_dim % 2:
        raise ValueError("period_len must be positive and token_dim positive/even")
    positions = torch.arange(period_len, dtype=torch.float64).unsqueeze(1)
    harmonics = torch.arange(
        1, token_dim // 2 + 1, dtype=torch.float64
    ).unsqueeze(0)
    angles = (2.0 * math.pi / period_len) * positions * harmonics
    features = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    return features.reshape(1, period_len, token_dim).float()


class GlucoScopeTokenizer(nn.Module):
    """Clip, fixed-normalize, and map each seven-quantile phase to one token."""

    def __init__(
        self,
        input_dim=7,
        token_dim=64,
        hidden_dim=64,
        zscore_mean=7.0,
        zscore_std=3.0,
        glucose_min=2.2,
        glucose_max=27.8,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.glucose_min = float(glucose_min)
        self.glucose_max = float(glucose_max)
        if zscore_std <= 0 or glucose_max <= glucose_min:
            raise ValueError("Invalid normalization or clipping range")
        self.register_buffer("zscore_mean", torch.tensor(float(zscore_mean)))
        self.register_buffer("zscore_std", torch.tensor(float(zscore_std)))
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        if x.dim() != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected (B,L,{self.input_dim}), got {tuple(x.shape)}")
        clipped = x.clamp(self.glucose_min, self.glucose_max)
        normalized = (clipped - self.zscore_mean.to(x)) / self.zscore_std.to(x)
        return self.norm(self.projection(normalized))


class GlucoScopeEncoder(nn.Module):
    """Routing encoder with configurable fixed Fourier position injection."""

    def __init__(
        self,
        period_len=96,
        token_dim=64,
        num_layers=2,
        num_routers=4,
        num_heads=4,
        dropout=0.1,
        position_dropout=0.0,
        position_injection="every_layer",
        position_encoding="fixed_fourier",
    ):
        super().__init__()
        self.period_len = int(period_len)
        self.token_dim = int(token_dim)
        self.position_injection = str(position_injection)
        if self.position_injection not in {"once", "every_layer"}:
            raise ValueError(
                "position_injection must be 'once' or 'every_layer'"
            )
        self.position_encoding_type = str(position_encoding)
        if self.position_encoding_type == "fixed_fourier":
            self.register_buffer(
                "position_encoding",
                build_glucoscope_fourier_encoding(period_len, token_dim),
                persistent=True,
            )
            self.position_embedding = None
        elif self.position_encoding_type == "learnable_embedding":
            self.register_buffer("position_encoding", None, persistent=False)
            self.position_embedding = nn.Embedding(period_len, token_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(
                "position_encoding must be 'fixed_fourier' or "
                "'learnable_embedding'"
            )
        self.position_dropout = nn.Dropout(float(position_dropout))
        self.routing_layers = nn.ModuleList(
            [
                GlucoScopeRoutingLayer(
                    latent_dim=token_dim,
                    num_routers=num_routers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, tokens):
        expected = (self.period_len, self.token_dim)
        if tokens.dim() != 3 or tokens.shape[1:] != expected:
            raise ValueError(f"Expected (B,{expected[0]},{expected[1]})")
        states = tokens.unsqueeze(1)
        if self.position_encoding_type == "fixed_fourier":
            positions = self.position_encoding.to(tokens)
        else:
            indices = torch.arange(self.period_len, device=tokens.device)
            positions = self.position_embedding(indices).unsqueeze(0).to(tokens.dtype)
        positions = positions.unsqueeze(1)
        if self.position_injection == "once":
            states = self.position_dropout(states + positions)
        for layer in self.routing_layers:
            if self.position_injection == "every_layer":
                states = self.position_dropout(states + positions)
            states = layer(states)
        return states.squeeze(1)


class GlucoScopeReconstructionHead(nn.Module):
    """Ordered seven-quantile decoder with level/shape structural separation."""

    def __init__(
        self,
        token_dim=64,
        hidden_dim=128,
        n_max=16,
        tau_prior=(2.0, 4.0, 7.0),
        learnable_tau=False,
        dropout=0.0,
        variant="level_shape",
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.n_max = int(n_max)
        self.variant = str(variant)
        if self.variant not in {"level_shape", "simple_phase_mlp"}:
            raise ValueError(
                "reconstruction variant must be 'level_shape' or "
                "'simple_phase_mlp'"
            )
        drop = (lambda: nn.Dropout(dropout)) if dropout > 0 else (lambda: nn.Identity())
        if self.variant == "simple_phase_mlp":
            self.phase_mlp = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                drop(),
            )
            self.median_head = nn.Sequential(
                nn.Linear(hidden_dim, token_dim),
                nn.GELU(),
                drop(),
                nn.Linear(token_dim, 1),
            )
            self.spread_head = nn.Sequential(
                nn.Linear(hidden_dim + 3, token_dim),
                nn.GELU(),
                drop(),
                nn.Linear(token_dim, 6),
            )
        else:
            self.median_head = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                drop(),
                nn.Linear(hidden_dim, 1),
            )
            # Phase features + 3 effective-day features + detached median.
            self.spread_head = nn.Sequential(
                nn.Linear(token_dim + 4, hidden_dim),
                nn.GELU(),
                drop(),
                nn.Linear(hidden_dim, 6),
            )
            self.level_head = nn.Linear(token_dim, 1)
            initial = torch.tensor(
                [math.log(math.expm1(float(value))) for value in tau_prior]
            )
            if learnable_tau:
                self.theta = nn.Parameter(initial)
            else:
                self.register_buffer("theta", initial)
        self.register_buffer(
            "gap_tier", torch.tensor([2, 1, 0, 0, 1, 2], dtype=torch.long)
        )

    @property
    def tau(self):
        return F.softplus(self.theta)

    def day_features(self, effective_days):
        days = effective_days.clamp(min=0).float()
        return torch.stack(
            [
                days / self.n_max,
                torch.log1p(days),
                1.0 / torch.sqrt(days.clamp(min=1.0)),
            ],
            dim=-1,
        )

    def reliability_gate(self, effective_days):
        elapsed = (effective_days.clamp(min=0).float() - 1.0).clamp(min=0)
        tier_gate = 1.0 - torch.exp(
            -elapsed.unsqueeze(-1) / self.tau.view(1, 1, 3)
        )
        return tier_gate[..., self.gap_tier]

    def forward(self, phase_states, global_state, effective_days):
        if effective_days.shape != phase_states.shape[:2]:
            raise ValueError("effective_days must have shape (B,96)")
        if self.variant == "simple_phase_mlp":
            phase_features = self.phase_mlp(phase_states)
            median = self.median_head(phase_features).squeeze(-1)
            level = median.mean(dim=1, keepdim=True)
            shape = median - level
            spread_input = torch.cat(
                [phase_features, self.day_features(effective_days)], dim=-1
            )
        else:
            raw_median = self.median_head(phase_states).squeeze(-1)
            shape = raw_median - raw_median.mean(dim=1, keepdim=True)
            level = self.level_head(global_state)
            median = level + shape
            spread_input = torch.cat(
                [phase_states, self.day_features(effective_days),
                 median.detach().unsqueeze(-1)],
                dim=-1,
            )
        raw_gaps = F.softplus(self.spread_head(spread_input))
        if self.variant == "simple_phase_mlp":
            gate = torch.ones_like(raw_gaps)
            gaps = raw_gaps
        else:
            gate = self.reliability_gate(effective_days)
            gaps = raw_gaps * gate
        ext_hi, tail_hi, core_hi, core_lo, tail_lo, ext_lo = gaps.unbind(-1)
        q75 = median + core_hi
        q95 = q75 + tail_hi
        maximum = q95 + ext_hi
        q25 = median - core_lo
        q5 = q25 - tail_lo
        minimum = q5 - ext_lo
        reconstruction = torch.stack(
            [maximum, q95, q75, median, q25, q5, minimum], dim=-1
        )
        return {
            "reconstruction": reconstruction,
            "level_prediction": level,
            "shape_prediction": shape,
            "median_prediction": median,
            "raw_gaps": raw_gaps,
            "gaps": gaps,
            "reliability_gate": gate,
        }


class GlucoScopeGlobalAttentionPooling(nn.Module):
    """Pretrained attention pooling used only for the backbone global feature."""

    def __init__(self, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, phase_states):
        weights = torch.softmax(self.score(phase_states).squeeze(-1), dim=1)
        return (phase_states * weights.unsqueeze(-1)).sum(dim=1), weights


class GlucoScope(nn.Module):
    """Canonical self-supervised GlucoScope V1 attention-pooling model."""

    MODEL_NAME = "GlucoScope"
    MODEL_VERSION = "V1"
    EXPECTED_PARAMETERS = 165_321

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.period_len = int(_cfg(config, "period_len", 96))
        self.pin = int(_cfg(config, "pin", 7))
        self.latent_dim = int(_cfg(config, "latent_dim", 64))
        self.tokenizer = GlucoScopeTokenizer(
            self.pin,
            self.latent_dim,
            int(_cfg(config, "tokenizer_hidden", 64)),
            float(_cfg(config, "zscore_mean", 7.0)),
            float(_cfg(config, "zscore_std", 3.0)),
            float(_cfg(config, "glucose_min", 2.2)),
            float(_cfg(config, "glucose_max", 27.8)),
        )
        self.masker = GlucoScopeMasker(self.latent_dim)
        self.encoder = GlucoScopeEncoder(
            self.period_len,
            self.latent_dim,
            int(_cfg(config, "phase_layers", 2)),
            int(_cfg(config, "phase_num_routers", 4)),
            int(_cfg(config, "phase_attn_heads", 4)),
            float(_cfg(config, "phase_attn_dropout", 0.1)),
            float(_cfg(config, "phase_pos_dropout", 0.0)),
            str(_cfg(config, "phase_position_injection", "every_layer")),
            str(_cfg(config, "phase_position_encoding", "fixed_fourier")),
        )
        self.global_pooling = GlucoScopeGlobalAttentionPooling(
            self.latent_dim,
            int(_cfg(config, "global_pooling_hidden", 128)),
        )
        self.reconstruction = GlucoScopeReconstructionHead(
            self.latent_dim,
            int(_cfg(config, "reconstruction_hidden", 128)),
            int(_cfg(config, "reconstruction_n_max", 16)),
            (
                float(_cfg(config, "tau_core", 2.0)),
                float(_cfg(config, "tau_tail", 4.0)),
                float(_cfg(config, "tau_extreme", 7.0)),
            ),
            bool(_cfg(config, "learnable_tau", False)),
            float(_cfg(config, "reconstruction_dropout", 0.0)),
            str(_cfg(config, "reconstruction_variant", "level_shape")),
        )

    def encode(self, x, phase_mask=None):
        if x.dim() != 3 or x.shape[1:] != (self.period_len, self.pin):
            raise ValueError(
                f"Expected (B,{self.period_len},{self.pin}), got {tuple(x.shape)}"
            )
        tokens = self.tokenizer(x)
        masked_tokens = self.masker(tokens, phase_mask)
        phase_states = self.encoder(masked_tokens)
        global_state, _ = self.global_pooling(phase_states)
        return phase_states, global_state

    def forward(self, x, phase_mask=None, phase_effective_days=None):
        if phase_effective_days is None:
            raise ValueError("phase_effective_days is required for reconstruction")
        phase_states, global_state = self.encode(x, phase_mask)
        output = self.reconstruction(
            phase_states, global_state, phase_effective_days
        )
        output.update(
            {
                "tokens": phase_states,
                "phase_representation": phase_states,
                "global_representation": global_state,
            }
        )
        return output


class GlucoScopeAttentionPooling(nn.Module):
    """Task-specific learned weights over the 96 phase representations."""

    def __init__(self, latent_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, phase_states):
        weights = torch.softmax(self.score(phase_states).squeeze(-1), dim=1)
        return (phase_states * weights.unsqueeze(-1)).sum(dim=1), weights


class GlucoScopeResidualHeadBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class GlucoScopeDownstream(nn.Module):
    """GlucoScope backbone, phase attention pooling, and clinical task head."""

    MODEL_NAME = "GlucoScope"

    def __init__(self, config, task_type, backbone=None):
        super().__init__()
        self.backbone = backbone or GlucoScope(config)
        self.task_type = str(task_type)
        latent = int(_cfg(config, "latent_dim", 64))
        hidden = int(_cfg(config, "head_hidden", 128))
        dropout = float(_cfg(config, "head_dropout", 0.1))
        self.pooling = GlucoScopeAttentionPooling(
            latent,
            int(_cfg(config, "pooling_hidden", 128)),
            float(_cfg(config, "pooling_dropout", 0.1)),
        )
        if self.task_type == "multiclass":
            output_dim = int(_cfg(config, "num_classes", 3))
            self.head = nn.Sequential(
                nn.LayerNorm(latent),
                nn.Linear(latent, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output_dim),
            )
        elif self.task_type in {"binary", "classification"}:
            self.head = nn.Sequential(
                nn.LayerNorm(latent),
                nn.Linear(latent, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
        elif self.task_type == "regression":
            blocks = int(_cfg(config, "head_layers", 2))
            modules = [
                nn.LayerNorm(latent),
                nn.Linear(latent, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            modules.extend(
                GlucoScopeResidualHeadBlock(hidden, dropout)
                for _ in range(blocks)
            )
            modules.extend([nn.LayerNorm(hidden), nn.Linear(hidden, 1)])
            self.head = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def encode_for_head(self, x):
        if x.dim() == 3:
            phase_states, _ = self.backbone.encode(x)
            return (*self.pooling(phase_states), phase_states)
        if x.dim() != 4:
            raise ValueError("Expected (B,96,7) or (B,V,96,7)")
        batch, views, phases, features = x.shape
        phase_states, _ = self.backbone.encode(
            x.reshape(batch * views, phases, features)
        )
        pooled, weights = self.pooling(phase_states)
        return (
            pooled.reshape(batch, views, -1).mean(dim=1),
            weights.reshape(batch, views, phases).mean(dim=1),
            phase_states.reshape(batch, views, phases, -1).mean(dim=1),
        )

    def forward(self, x, return_features=False):
        pooled, weights, phase_states = self.encode_for_head(x)
        prediction = self.head(pooled)
        if prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        if return_features:
            return {
                "prediction": prediction,
                "pooled_representation": pooled,
                "phase_representation": phase_states,
                "attention_weights": weights,
            }
        return prediction
