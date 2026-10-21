"""Router attention used by the GlucoScope phase encoder."""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn


class GlucoScopeFullAttention(nn.Module):
    """Non-causal scaled dot-product attention on projected multi-head tensors."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, queries, keys, values):
        # queries (B,L,H,E), keys/values (B,S,H,E)
        scale = 1.0 / sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        weights = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", weights, values)
        return output.contiguous()


class GlucoScopeAttentionLayer(nn.Module):
    """Linear Q/K/V projections around :class:`GlucoScopeFullAttention`."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        head_dim = model_dim // num_heads
        self.num_heads = int(num_heads)
        self.query_projection = nn.Linear(model_dim, head_dim * num_heads)
        self.key_projection = nn.Linear(model_dim, head_dim * num_heads)
        self.value_projection = nn.Linear(model_dim, head_dim * num_heads)
        self.out_projection = nn.Linear(head_dim * num_heads, model_dim)
        self.inner_attention = GlucoScopeFullAttention(dropout)

    def forward(self, queries, keys, values):
        batch, query_len, _ = queries.shape
        source_len = keys.shape[1]
        heads = self.num_heads
        q = self.query_projection(queries).view(batch, query_len, heads, -1)
        k = self.key_projection(keys).view(batch, source_len, heads, -1)
        v = self.value_projection(values).view(batch, source_len, heads, -1)
        attended = self.inner_attention(q, k, v).view(batch, query_len, -1)
        return self.out_projection(attended)


class GlucoScopeRoutingLayer(nn.Module):
    """Two-stage router aggregation/distribution with residual MLP.

    Input and output are ``(B, 1, 96, 64)`` in the canonical configuration.
    Four learned routers first read all phase tokens, then phase tokens read the
    router buffer. This retains the exact V1 parameterization.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_routers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_routers = int(num_routers)
        self.router = nn.Parameter(torch.empty(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)
        self.router_sender = GlucoScopeAttentionLayer(
            latent_dim, num_heads, dropout
        )
        self.router_receiver = GlucoScopeAttentionLayer(
            latent_dim, num_heads, dropout
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, states):
        if states.dim() != 4 or states.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected (B,C,L,{self.latent_dim}), got {tuple(states.shape)}"
            )
        batch, channels, phase_len, latent_dim = states.shape
        tokens = states.reshape(batch * channels, phase_len, latent_dim)
        routers = self.router.unsqueeze(0).expand(batch * channels, -1, -1)
        router_buffer = self.router_sender(routers, tokens, tokens)
        received = self.router_receiver(tokens, router_buffer, router_buffer)
        output = self.norm1(tokens + self.dropout_layer(received))
        output = self.norm2(output + self.dropout_layer(self.mlp(output)))
        return output.reshape(batch, channels, phase_len, latent_dim)
