"""Self-contained GlucoScope pretraining engine."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from layers.glucoscope import GlucoScope
from layers.losses import (
    GlucoScopePretrainingLoss,
    calibrate_glucoscope_scales,
)
from layers.masking import GlucoScopeMaskGenerator
from provider.pretrain_dataset import (
    GlucoScopePretrainDataset,
    GlucoScopeMaterializedPretrainDataset,
    split_pretraining_dataset,
)
from utils.checkpoint import load_backbone, load_checkpoint, parameter_count
from utils.config import project_path, save_config
from utils.reproducibility import select_device, set_seed
from utils.visualization import plot_history


def _mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def _selected_mean(values, mask):
    weights = mask.to(values)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1)

def _reconstruction_selection_score(metrics, weights):
    names = (
        "full_all_quantile_mae",
        "masked_median_mae",
        "masked_width_mae",
        "masked_slope_rmse",
    )
    configured = {name: float(weights.get(name, 0.0)) for name in names}
    weight_sum = sum(configured.values())
    if weight_sum <= 0:
        raise ValueError("At least one reconstruction selection weight must be positive")
    return sum(
        configured[name] * float(metrics[name]) for name in names
    ) / weight_sum



def run_epoch(model, loader, criterion, masks, device, optimizer=None, seed=0):
    training = optimizer is not None
    model.train(training)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        days = batch["phase_effective_days"].to(device)
        raw_days = batch["raw_day_count"].to(device)
        phase_mask, strategies = masks(
            x.shape[0],
            x.shape[1],
            device,
            effective_days=raw_days,
            generator=generator,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(x, phase_mask, days)
            loss, components = criterion(output, x, phase_mask)
            if training:
                loss.backward()
                optimizer.step()
        target = x.clamp(criterion.glucose_min, criterion.glucose_max)
        median_error = (output["median_prediction"] - target[..., 3]).abs()
        absolute_error = (output["reconstruction"] - target).abs()
        width_error = (
            (output["reconstruction"][..., 0] - output["reconstruction"][..., 6])
            - (target[..., 0] - target[..., 6])
        ).abs()
        slope_pred = torch.roll(output["median_prediction"], -1, 1) - output[
            "median_prediction"
        ]
        slope_true = torch.roll(target[..., 3], -1, 1) - target[..., 3]
        edge_mask = phase_mask | torch.roll(phase_mask, -1, 1)
        rows.append(
            {
                "loss": float(loss.detach()),
                "slope": float(components["slope"]),
                "reconstruction": float(components["reconstruction"]),
                "masked_reconstruction": float(
                    components["masked_reconstruction"]
                ),
                "visible_reconstruction": float(
                    components["visible_reconstruction"]
                ),
                "extreme_fraction": float(components["extreme_fraction"]),
                "full_all_quantile_mae": float(absolute_error.mean().detach()),
                "masked_all_quantile_mae": float(
                    _selected_mean(absolute_error, phase_mask).detach()
                ),
                "masked_width_mae": float(
                    _selected_mean(width_error, phase_mask).detach()
                ),
                "mask_rate": float(phase_mask.float().mean()),
                "point_rate": float((strategies == 0).float().mean()),
                "multi_block_rate": float((strategies == 1).float().mean()),
                "single_block_rate": float((strategies == 2).float().mean()),
                "mixed_rate": float((strategies == 3).float().mean()),
                "masked_median_mae": float(
                    _selected_mean(median_error, phase_mask).detach()
                ),
                "masked_slope_rmse": float(
                    _selected_mean(
                        (slope_pred - slope_true).square(), edge_mask
                    ).sqrt().detach()
                ),
            }
        )
    if not rows:
        raise ValueError("No batches were processed")
    return {key: _mean(rows, key) for key in rows[0]}


def _epoch_learning_rate(epoch, train_config):
    base = float(train_config["learning_rate"])
    minimum = float(train_config.get("minimum_learning_rate", 0.0))
    warmup = int(train_config.get("warmup_epochs", 0))
    total = int(train_config["epochs"])
    if warmup > 0 and epoch <= warmup:
        return base * epoch / warmup
    if train_config.get("scheduler", "cosine") != "cosine":
        return base
    progress = (epoch - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def run_pretraining(config):
    model_config = config["model"]
    model_version = str(model_config.get("version", GlucoScope.MODEL_VERSION))
    data_config = config["data"]
    train_config = config["train"]
    loss_config = config["loss"]
    mask_config = config["mask"]
    output_dir = project_path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(train_config["seed"], train_config.get("deterministic", True))
    device = select_device(train_config.get("use_gpu", True), train_config.get("gpu", 0))

    if data_config.get("materialized_path"):
        dataset = GlucoScopeMaterializedPretrainDataset(
            project_path(data_config["materialized_path"]),
            data_config.get("min_day_count", 1.0),
            data_config.get("strict_quantile_order", False),
        )
    else:
        dataset = GlucoScopePretrainDataset(
            project_path(data_config["data_path"]),
            project_path(data_config["remark_path"]),
            data_config.get("day_count_column", "effective_days"),
            data_config.get("min_day_count", 1.0),
            data_config.get("strict_quantile_order", False),
        )
    training, validation = split_pretraining_dataset(
        dataset,
        data_config.get("validation_ratio", 0.1),
        train_config["seed"],
    )
    loader_kwargs = {
        "batch_size": train_config["batch_size"],
        "num_workers": train_config.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        training,
        shuffle=train_config.get("shuffle", True),
        **loader_kwargs,
    )
    validation_loader = DataLoader(validation, shuffle=False, **loader_kwargs)
    selection_size = min(
        len(validation), int(train_config.get("selection_max_samples", 2048))
    )
    selection_loader = DataLoader(
        Subset(validation, list(range(selection_size))),
        shuffle=False,
        **loader_kwargs,
    )
    calibration_loader = DataLoader(
        training,
        batch_size=max(1024, train_config["batch_size"]),
        shuffle=False,
        num_workers=train_config.get("num_workers", 0),
    )
    scales = calibrate_glucoscope_scales(
        calibration_loader,
        model_config["glucose_min"],
        model_config["glucose_max"],
        loss_config["calibration_min_days"],
        loss_config["slope_sigma_floor"],
    )
    model = GlucoScope(model_config).to(device)
    actual_parameters = parameter_count(model)
    expected = int(config["model"].get("expected_parameter_count", GlucoScope.EXPECTED_PARAMETERS))
    if actual_parameters != expected:
        raise AssertionError(
            f"GlucoScope {model_version} parameter count "
            f"{actual_parameters} != {expected}"
        )
    initial_checkpoint = config.get("initial_checkpoint")
    resume_checkpoint = config.get("resume_checkpoint")
    if initial_checkpoint and resume_checkpoint:
        raise ValueError(
            "initial_checkpoint and resume_checkpoint are mutually exclusive"
        )
    if initial_checkpoint:
        initial_checkpoint = project_path(initial_checkpoint)
        source_checkpoint = load_backbone(model, initial_checkpoint, strict=True)
        source_parameters = int(
            source_checkpoint.get("parameter_count", actual_parameters)
        )
        if source_parameters != actual_parameters:
            raise ValueError(
                "Initial checkpoint parameter count does not match the model"
            )
    criterion = GlucoScopePretrainingLoss(
        scales["sigma_slope"],
        slope_weight=loss_config["slope_weight"],
        visible_loss_weight=loss_config["visible_loss_weight"],
        reconstruction_weight=loss_config["reconstruction_weight"],
        hypoglycemia_threshold=loss_config["hypoglycemia_threshold"],
        hyperglycemia_threshold=loss_config["hyperglycemia_threshold"],
        extreme_glucose_weight=loss_config["extreme_glucose_weight"],
        beta=loss_config["huber_beta"],
        glucose_min=model_config["glucose_min"],
        glucose_max=model_config["glucose_max"],
    ).to(device)
    masks = GlucoScopeMaskGenerator(**mask_config)
    unmasked_masks = GlucoScopeMaskGenerator(
        **{**mask_config, "ratio_min": 0.0, "ratio_max": 0.0}
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    resume_epoch = 0
    resume_learning_rate = None
    if resume_checkpoint:
        resume_checkpoint = project_path(resume_checkpoint)
        source_checkpoint, source_state = load_checkpoint(resume_checkpoint)
        model.load_state_dict(source_state, strict=True)
        source_parameters = int(
            source_checkpoint.get("parameter_count", actual_parameters)
        )
        if source_parameters != actual_parameters:
            raise ValueError(
                "Resume checkpoint parameter count does not match the model"
            )
        if "optimizer_state_dict" not in source_checkpoint:
            raise ValueError("Resume checkpoint has no optimizer state")
        optimizer.load_state_dict(source_checkpoint["optimizer_state_dict"])
        if source_checkpoint.get("loss_state_dict"):
            criterion.load_state_dict(source_checkpoint["loss_state_dict"])
        resume_epoch = int(source_checkpoint.get("epoch", 0))
        resume_learning_rate = float(optimizer.param_groups[0]["lr"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = resume_learning_rate
        print(
            f"Resuming from {resume_checkpoint} at epoch={resume_epoch} "
            f"lr={resume_learning_rate:.8g}",
            flush=True,
        )
    save_config(config, output_dir / "resolved_config.yaml")
    (output_dir / "data_audit.json").write_text(
        json.dumps(dataset.audit(), indent=2), encoding="utf-8"
    )
    writer = (
        SummaryWriter(output_dir / "tensorboard")
        if train_config.get("tensorboard", True)
        else None
    )
    history, best_loss, best_score, best_epoch, stale = (
        [],
        float("inf"),
        float("inf"),
        None,
        0,
    )
    milestones = set(train_config.get("milestone_epochs", []))
    schedule_config = dict(train_config)
    if resume_learning_rate is not None:
        schedule_config["learning_rate"] = resume_learning_rate
        schedule_config["warmup_epochs"] = 0
    for stage_epoch in range(1, train_config["epochs"] + 1):
        epoch = resume_epoch + stage_epoch
        current_lr = _epoch_learning_rate(stage_epoch, schedule_config)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            masks,
            device,
            optimizer,
            train_config["seed"] + epoch,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            masks,
            device,
            seed=train_config["seed"] + 10_000,
        )
        unmasked_metrics = run_epoch(
            model,
            selection_loader,
            criterion,
            unmasked_masks,
            device,
            seed=train_config["seed"] + 20_000,
        )
        selection_metrics = dict(validation_metrics)
        selection_metrics["full_all_quantile_mae"] = (
            unmasked_metrics["full_all_quantile_mae"]
        )
        selection_score = _reconstruction_selection_score(
            selection_metrics,
            train_config.get(
                "selection_weights",
                {
                    "full_all_quantile_mae": 0.35,
                    "masked_median_mae": 0.35,
                    "masked_width_mae": 0.15,
                    "masked_slope_rmse": 0.15,
                },
            ),
        )
        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "selection_score": selection_score,
            "train": train_metrics,
            "validation": validation_metrics,
            "unmasked_validation": {
                "samples": selection_size,
                "full_all_quantile_mae": unmasked_metrics["full_all_quantile_mae"],
            },
        }
        history.append(row)
        payload = {
            "model_name": "GlucoScope",
            "model_version": model_version,
            "variant": "A",
            "initial_checkpoint": (
                str(initial_checkpoint) if initial_checkpoint else None
            ),
            "resume_checkpoint": (
                str(resume_checkpoint) if resume_checkpoint else None
            ),
            "stage_epoch": stage_epoch,
            "model_state_dict": model.state_dict(),
            "loss_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "parameter_count": actual_parameters,
            "loss_scales": scales,
            "config": config,
            "metrics": row,
        }
        best_loss = min(best_loss, validation_metrics["loss"])
        improved = selection_score < (
            best_score - train_config.get("min_delta", 0.0)
        )
        if improved:
            best_score, best_epoch, stale = selection_score, epoch, 0
            torch.save(payload, output_dir / "best.pth")
        else:
            stale += 1
        if epoch in milestones:
            torch.save(payload, output_dir / f"epoch_{epoch:04d}.pth")
        torch.save(payload, output_dir / "last.pth")
        if writer is not None:
            for split, metrics in (
                ("train", train_metrics), ("validation", validation_metrics)
            ):
                for name, value in metrics.items():
                    writer.add_scalar(f"{split}/{name}", value, epoch)
            writer.add_scalar("selection/score", selection_score, epoch)
            writer.add_scalar(
                "selection/unmasked_full_all_quantile_mae",
                unmasked_metrics["full_all_quantile_mae"], epoch,
            )
            writer.add_scalar("train/learning_rate", current_lr, epoch)
            writer.flush()
        print(
            f"GlucoScope epoch={epoch} lr={current_lr:.8g} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} "
            f"reconstruction_score={selection_score:.6f}",
            flush=True,
        )
        summary = {
            "model_name": "GlucoScope",
            "model_version": model_version,
            "parameter_count": actual_parameters,
            "best_validation_loss": best_loss,
            "best_reconstruction_score": best_score,
            "best_epoch": best_epoch,
            "resume_checkpoint": (
                str(resume_checkpoint) if resume_checkpoint else None
            ),
            "resume_epoch": resume_epoch,
            "loss_scales": scales,
            "history": history,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if (
            stale >= train_config["patience"]
            and epoch >= train_config.get("minimum_epochs", 0)
        ):
            break
    if writer is not None:
        writer.close()
    plot_history(history, output_dir / "loss_curve.png")
    return summary
