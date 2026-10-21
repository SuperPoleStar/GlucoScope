"""Calibrated masked quantile reconstruction and cyclic-slope objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _selected_mean(values, selection):
    weights = selection.to(values)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def calibrate_glucoscope_scales(
    loader,
    glucose_min=2.2,
    glucose_max=27.8,
    minimum_effective_days=8.0,
    slope_floor=0.10,
):
    """Fit only target scales, using eligible samples from the training split."""
    slopes = []
    sample_count = 0
    for batch in loader:
        target = batch["x"].float().clamp(glucose_min, glucose_max)
        days = batch["raw_day_count"].reshape(-1).float()
        eligible = days >= float(minimum_effective_days)
        if not bool(eligible.any()):
            continue
        median = target[eligible, :, 3]
        slope = torch.roll(median, -1, 1) - median
        slopes.append(slope.flatten())
        sample_count += int(eligible.sum())
    if not slopes:
        raise ValueError("No training samples meet calibration_min_days")
    sigma_slope_raw = torch.cat(slopes).std(unbiased=False).item()
    return {
        "sigma_slope": max(float(slope_floor), sigma_slope_raw),
        "sigma_slope_raw": sigma_slope_raw,
        "calibration_samples": sample_count,
        "minimum_effective_days": float(minimum_effective_days),
    }


class GlucoScopePretrainingLoss(nn.Module):
    """Masked quantile reconstruction plus cyclic median-slope supervision."""

    def __init__(
        self,
        sigma_slope,
        slope_weight=2.0,
        visible_loss_weight=0.1,
        reconstruction_weight=1.0,
        hypoglycemia_threshold=3.9,
        hyperglycemia_threshold=10.0,
        extreme_glucose_weight=2.0,
        beta=1.0,
        glucose_min=2.2,
        glucose_max=27.8,
    ):
        super().__init__()
        if float(sigma_slope) <= 0:
            raise ValueError("sigma_slope must be positive")
        self.register_buffer("sigma_slope", torch.tensor(float(sigma_slope)))
        self.slope_weight = float(slope_weight)
        self.visible_loss_weight = float(visible_loss_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.hypoglycemia_threshold = float(hypoglycemia_threshold)
        self.hyperglycemia_threshold = float(hyperglycemia_threshold)
        self.extreme_glucose_weight = float(extreme_glucose_weight)
        if self.reconstruction_weight < 0 or self.extreme_glucose_weight < 1:
            raise ValueError("loss weights must be nonnegative and extreme weight >= 1")
        self.beta = float(beta)
        self.glucose_min = float(glucose_min)
        self.glucose_max = float(glucose_max)

    def forward(self, output, target, ssl_mask):
        target = target.clamp(self.glucose_min, self.glucose_max)
        median = target[..., 3]
        slope_target = torch.roll(median, -1, 1) - median
        median_prediction = output["median_prediction"]
        slope_prediction = torch.roll(median_prediction, -1, 1) - median_prediction
        slope_loss = F.smooth_l1_loss(
            (slope_prediction - slope_target) / self.sigma_slope,
            torch.zeros_like(slope_target),
            beta=self.beta,
            reduction="none",
        )
        reconstruction_element = F.smooth_l1_loss(
            output["reconstruction"], target, beta=self.beta, reduction="none"
        )
        extreme_phase = (median < self.hypoglycemia_threshold) | (
            median > self.hyperglycemia_threshold
        )
        phase_weight = torch.where(
            extreme_phase,
            torch.full_like(median, self.extreme_glucose_weight),
            torch.ones_like(median),
        )
        reconstruction_loss = (
            reconstruction_element * phase_weight.unsqueeze(-1)
        ).mean(dim=-1)
        masked = ssl_mask.bool()
        visible = ~masked
        masked_edges = masked | torch.roll(masked, -1, 1)
        visible_edges = ~masked_edges
        masked_reconstruction = _selected_mean(reconstruction_loss, masked)
        visible_reconstruction = _selected_mean(reconstruction_loss, visible)
        masked_slope = _selected_mean(slope_loss, masked_edges)
        visible_slope = _selected_mean(slope_loss, visible_edges)
        weighted_reconstruction = masked_reconstruction + (
            self.visible_loss_weight * visible_reconstruction
        )
        weighted_slope = masked_slope + self.visible_loss_weight * visible_slope
        total = (
            self.reconstruction_weight * weighted_reconstruction
            + self.slope_weight * weighted_slope
        )
        components = {
            "slope": slope_loss.mean().detach(),
            "reconstruction": reconstruction_loss.mean().detach(),
            "masked_reconstruction": masked_reconstruction.detach(),
            "visible_reconstruction": visible_reconstruction.detach(),
            "extreme_fraction": extreme_phase.float().mean().detach(),
            "masked": masked_slope.detach(),
            "visible": visible_slope.detach(),
        }
        return total, components
