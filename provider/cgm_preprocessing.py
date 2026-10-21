"""Convert raw CGM records into complete 96 x 7 GlucoScope AGP tables."""

from __future__ import annotations

import bisect
import concurrent.futures
import csv
import hashlib
import json
import math
import random
import time as wall_time
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PHASES = 96
PHASE_MINUTES = 15
SIGMA_MINUTES = 5.0
CUTOFF_MINUTES = 15.0
GLUCOSE_MIN = 2.2
GLUCOSE_MAX = 27.8
GAUSSIAN_ROUND_DECIMALS = 1
QUANTILES = (1.0, 0.95, 0.75, 0.5, 0.25, 0.05, 0.0)
FEATURE_NAMES = ("max", "p95", "p75", "median", "p25", "p5", "min")
TIME_NAMES = {"time", "timestamp", "datetime", "date_time", "bgdate", "时间"}
GLUCOSE_NAMES = {
    "glucose",
    "cgm",
    "value",
    "bg",
    "sg",
    "glucose value(mmol/l)",
    "葡萄糖历史记录（mmol/l）",
    "葡萄糖历史记录(mmol/l)",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_time(value):
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.to_pydatetime()


def _clip_glucose(values):
    return np.clip(values, GLUCOSE_MIN, GLUCOSE_MAX)


def _round_half_up_positive(value, decimals=GAUSSIAN_ROUND_DECIMALS):
    """Decimal half-up rounding for positive glucose values."""
    factor = 10 ** int(decimals)
    return math.floor(float(value) * factor + 0.5) / factor


def read_subject_csv(path):
    """Read common one- or two-line-header subject CGM files."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    for header_index, row in enumerate(rows[:10]):
        normalized = [str(value).strip().lower() for value in row]
        time_index = next(
            (index for index, value in enumerate(normalized) if value in TIME_NAMES),
            None,
        )
        glucose_index = next(
            (
                index
                for index, value in enumerate(normalized)
                if value in GLUCOSE_NAMES
            ),
            None,
        )
        if time_index is None or glucose_index is None:
            continue
        records = []
        for values in rows[header_index + 1 :]:
            if len(values) <= max(time_index, glucose_index):
                continue
            timestamp = _parse_time(values[time_index])
            try:
                glucose = float(values[glucose_index])
            except (TypeError, ValueError):
                continue
            if timestamp is not None and np.isfinite(glucose) and glucose > 0:
                records.append((timestamp, glucose))
        return sorted(records)
    raise ValueError("Could not find time and glucose columns")


def _complete_day(records):
    phases = {
        min(
            PHASES - 1,
            int(
                (
                    timestamp.hour * 60
                    + timestamp.minute
                    + timestamp.second / 60
                )
                // PHASE_MINUTES
            ),
        )
        for timestamp, _ in records
    }
    return len(phases) == PHASES


def aggregate_subject(
    records, sigma_minutes=SIGMA_MINUTES, cutoff_minutes=CUTOFF_MINUTES
):
    """Gaussian-align raw records, round daily phases, then aggregate."""
    sigma_minutes = float(sigma_minutes)
    cutoff_minutes = float(cutoff_minutes)
    if sigma_minutes <= 0 or cutoff_minutes <= 0:
        raise ValueError("Gaussian sigma and cutoff must be positive")
    by_date = defaultdict(list)
    for timestamp, glucose in records:
        if isinstance(timestamp, datetime) and np.isfinite(glucose) and glucose > 0:
            by_date[timestamp.date()].append(
                (timestamp, float(_clip_glucose(glucose)))
            )
    dates = sorted(by_date)
    if not dates:
        raise ValueError("No valid positive timestamped CGM records")
    for date in dates:
        by_date[date].sort()
    start = 1 if not _complete_day(by_date[dates[0]]) else 0
    end = len(dates) - (1 if len(dates) > 1 and not _complete_day(by_date[dates[-1]]) else 0)
    statistical_dates = dates[start:end]
    if not statistical_dates:
        raise ValueError("No statistical days after boundary-day filtering")
    values_by_phase = [[] for _ in range(PHASES)]
    for current_date in statistical_dates:
        current_records = by_date[current_date]
        current_times = [item[0] for item in current_records]
        previous = by_date.get(current_date - timedelta(days=1), [])
        previous_last = previous[-1] if previous else None
        for phase in range(PHASES):
            center = datetime.combine(current_date, time()) + timedelta(
                minutes=phase * PHASE_MINUTES
            )
            lower, upper = (
                center - timedelta(minutes=cutoff_minutes),
                center + timedelta(minutes=cutoff_minutes),
            )
            left = bisect.bisect_right(current_times, lower)
            right = bisect.bisect_left(current_times, upper)
            candidates = list(current_records[left:right])
            if phase == 0 and previous_last is not None:
                candidates.append(previous_last)
            weighted_sum, total_weight = 0.0, 0.0
            for timestamp, glucose in candidates:
                distance = abs((timestamp - center).total_seconds()) / 60
                if distance >= cutoff_minutes:
                    continue
                weight = math.exp(-0.5 * (distance / sigma_minutes) ** 2)
                weighted_sum += weight * glucose
                total_weight += weight
            if total_weight > 1e-8:
                aligned = weighted_sum / total_weight
                values_by_phase[phase].append(_round_half_up_positive(aligned))
    phase_days = np.asarray([len(values) for values in values_by_phase], dtype=int)
    if np.any(phase_days == 0):
        raise ValueError(
            f"Missing aligned values at phases {np.flatnonzero(phase_days == 0).tolist()}"
        )
    features = np.stack(
        [np.quantile(np.asarray(values), QUANTILES) for values in values_by_phase]
    )
    return {
        "features": features,
        "phase_effective_days": phase_days,
        "effective_days": int(phase_days.min()),
        "monitoring_days": len(dates),
        "natural_days": len(dates),
        "statistical_days": len(statistical_dates),
    }


def _format_phase(values):
    return "{" + ",".join(f"{float(value):.10g}" for value in values) + "}"


def write_agp_dataset(output_dir, included, excluded, provenance):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "agp.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID"] + [f"phase_{index}" for index in range(PHASES)])
        for sample_id, result in included:
            writer.writerow(
                [sample_id]
                + [_format_phase(values) for values in result["features"]]
            )
    with (output_dir / "remark.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["ID", "effective_days", "monitoring_days"]
            + [f"phase_{index}_effective_days" for index in range(PHASES)]
        )
        for sample_id, result in included:
            writer.writerow(
                [
                    sample_id,
                    result["effective_days"],
                    result["monitoring_days"],
                    *result["phase_effective_days"].tolist(),
                ]
            )
    fixed_width = provenance.get("phase_selection") == "first_point"
    manifest = {
        "model_name": "GlucoScope",
        "model_version": "V1",
        "source": provenance,
        "included": len(included),
        "excluded": excluded,
        "phases": PHASES,
        "phase_minutes": PHASE_MINUTES,
        "features": list(FEATURE_NAMES),
        "quantiles": list(QUANTILES),
        "alignment_method": (
            "fixed_phase_first_point" if fixed_width else "timestamp_gaussian"
        ),
        "gaussian_sigma_minutes": (
            None if fixed_width else provenance.get(
                "gaussian_sigma_minutes", SIGMA_MINUTES
            )
        ),
        "open_alignment_cutoff_minutes": (
            None if fixed_width else provenance.get(
                "open_alignment_cutoff_minutes", CUTOFF_MINUTES
            )
        ),
        "gaussian_round_decimals": (
            None if fixed_width else GAUSSIAN_ROUND_DECIMALS
        ),
        "gaussian_rounding_method": None if fixed_width else "half_up",
        "glucose_clip_min_mmol_l": GLUCOSE_MIN,
        "glucose_clip_max_mmol_l": GLUCOSE_MAX,
        "effective_days_definition": "minimum valid aligned days across 96 phases",
        "monitoring_days_definition": (
            "complete fixed-width day blocks"
            if fixed_width
            else "natural covered days before boundary-day filtering"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_from_directory(
    input_dir, output_dir, sigma_minutes=SIGMA_MINUTES,
    cutoff_minutes=CUTOFF_MINUTES,
):
    """Process one raw CGM CSV per subject; filename stem is the ID."""
    input_dir = Path(input_dir)
    included, excluded = [], []
    for path in sorted(input_dir.glob("*.csv")):
        try:
            included.append((
                path.stem,
                aggregate_subject(
                    read_subject_csv(path), sigma_minutes, cutoff_minutes
                ),
            ))
        except (OSError, ValueError) as error:
            excluded.append({"ID": path.stem, "reason": str(error)})
    return write_agp_dataset(
        output_dir,
        included,
        excluded,
        {
            "format": "subject_csv_directory",
            "path": str(input_dir.resolve()),
            "files": len(list(input_dir.glob("*.csv"))),
            "gaussian_sigma_minutes": float(sigma_minutes),
            "open_alignment_cutoff_minutes": float(cutoff_minutes),
        },
    )


def _process_subject_path(item):
    path, sigma_minutes, cutoff_minutes = item
    path = Path(path)
    try:
        return path.stem, aggregate_subject(
            read_subject_csv(path), sigma_minutes, cutoff_minutes
        ), None
    except (OSError, ValueError) as error:
        return path.stem, None, str(error)


def _duration_text(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_from_directories(
    input_dirs,
    output_dir,
    max_files_per_directory=None,
    sample_seed=42,
    num_workers=1,
    progress_every=1000,
    sigma_minutes=SIGMA_MINUTES,
    cutoff_minutes=CUTOFF_MINUTES,
):
    """Process multiple subject-CSV directories into one ID-unique dataset."""
    included, excluded, sources, selected_paths = [], [], [], []
    seen_ids = set()
    sampler = random.Random(int(sample_seed))
    for input_dir in [Path(value) for value in input_dirs]:
        all_paths = sorted(input_dir.glob("*.csv"))
        paths = all_paths
        if max_files_per_directory is not None:
            count = min(int(max_files_per_directory), len(all_paths))
            paths = sorted(sampler.sample(all_paths, count))
        source = {
            "path": str(input_dir.resolve()),
            "available_files": len(all_paths),
            "selected_files": len(paths),
        }
        if max_files_per_directory is not None:
            source["selected_ids"] = [path.stem for path in paths]
        sources.append(source)
        for path in paths:
            sample_id = path.stem
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate ID across input directories: {sample_id}")
            seen_ids.add(sample_id)
            selected_paths.append(path)
    workers = max(1, int(num_workers))
    every = max(1, int(progress_every))
    total = len(selected_paths)
    started = wall_time.monotonic()
    work_items = [
        (path, float(sigma_minutes), float(cutoff_minutes))
        for path in selected_paths
    ]
    if workers == 1:
        results = map(_process_subject_path, work_items)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_process_subject_path, work_items, chunksize=16)
    try:
        for completed, (sample_id, result, error) in enumerate(results, start=1):
            if error is None:
                included.append((sample_id, result))
            else:
                excluded.append({"ID": sample_id, "reason": error})
            if completed == 1 or completed % every == 0 or completed == total:
                elapsed = wall_time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (total - completed) / max(rate, 1e-9)
                print(
                    f"[preprocess] {completed}/{total} "
                    f"({100.0 * completed / max(total, 1):.2f}%) "
                    f"rate={rate:.2f} samples/s "
                    f"elapsed={_duration_text(elapsed)} "
                    f"eta={_duration_text(remaining)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return write_agp_dataset(
        output_dir,
        included,
        excluded,
        {
            "format": "subject_csv_directories",
            "directories": sources,
            "num_workers": workers,
            "elapsed_seconds": wall_time.monotonic() - started,
            "gaussian_sigma_minutes": float(sigma_minutes),
            "open_alignment_cutoff_minutes": float(cutoff_minutes),
        },
    )


def _aggregate_fixed_subject(item):
    sample_id, subject = item
    counts = np.isfinite(subject).sum(axis=0)
    if np.any(counts == 0):
        return sample_id, None, "phase without valid values"
    return sample_id, {
        "features": np.nanquantile(subject, QUANTILES, axis=0).T,
        "phase_effective_days": counts,
        "effective_days": int(counts.min()),
        "monitoring_days": int(subject.shape[0]),
    }, None


def _aggregate_fixed_values(ids, values, num_workers=1, progress_every=1000):
    items = list(zip([str(value) for value in ids], values))
    workers = max(1, int(num_workers))
    every = max(1, int(progress_every))
    total = len(items)
    started = wall_time.monotonic()
    executor = None
    if workers == 1:
        results = map(_aggregate_fixed_subject, items)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        results = executor.map(_aggregate_fixed_subject, items)
    included, excluded = [], []
    try:
        for completed, (sample_id, result, error) in enumerate(results, start=1):
            if error is None:
                included.append((sample_id, result))
            else:
                excluded.append({"ID": sample_id, "reason": error})
            if completed == 1 or completed % every == 0 or completed == total:
                elapsed = wall_time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (total - completed) / max(rate, 1e-9)
                print(
                    f"[preprocess] {completed}/{total} "
                    f"({100.0 * completed / max(total, 1):.2f}%) "
                    f"rate={rate:.2f} samples/s "
                    f"elapsed={_duration_text(elapsed)} "
                    f"eta={_duration_text(remaining)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return included, excluded, wall_time.monotonic() - started


def build_from_wide(
    input_path, output_dir, block_width=98, num_workers=1, progress_every=1000
):
    """Process ID plus repeated DAY_ID, Date, fixed-width CGM day blocks."""
    input_path = Path(input_path)
    table = pd.read_csv(input_path, low_memory=False)
    block_width = int(block_width)
    day_points = block_width - 2
    if (
        table.columns[0] != "ID"
        or (table.shape[1] - 1) % block_width
        or day_points % PHASES
    ):
        raise ValueError(
            "Wide input must be ID plus complete DAY_ID, Date, CGM day blocks "
            "whose point count is divisible by 96"
        )
    if table.iloc[:, 0].astype(str).duplicated().any():
        raise ValueError("Wide input IDs must be unique")
    days = (table.shape[1] - 1) // block_width
    points_per_phase = day_points // PHASES
    daily = []
    for day in range(days):
        start = 1 + day * block_width + 2
        raw_day = table.iloc[:, start : start + day_points].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        raw_day[raw_day <= 0] = np.nan
        raw_day = _clip_glucose(raw_day)
        dense = raw_day.reshape(len(table), PHASES, points_per_phase)
        daily.append(dense[..., 0])
    values = np.stack(daily, axis=1)
    included, excluded, elapsed = _aggregate_fixed_values(
        table.iloc[:, 0].astype(str), values, num_workers, progress_every
    )
    return write_agp_dataset(
        output_dir,
        included,
        excluded,
        {
            "format": "wide_day_blocks",
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "day_blocks": days,
            "day_points": day_points,
            "points_per_phase": points_per_phase,
            "phase_selection": "first_point",
            "num_workers": max(1, int(num_workers)),
            "elapsed_seconds": elapsed,
        },
    )


def build_from_dense_wide(
    input_path, output_dir, day_points=288, metadata_columns=1,
    num_workers=1, progress_every=1000,
    sigma_minutes=SIGMA_MINUTES, cutoff_minutes=CUTOFF_MINUTES,
):
    """Process ID + metadata + consecutive fixed-width dense CGM day blocks."""
    input_path = Path(input_path)
    table = pd.read_csv(input_path)
    if table.columns[0] != "ID":
        raise ValueError("Dense-wide input must start with ID")
    if table.iloc[:, 0].astype(str).duplicated().any():
        raise ValueError("Dense-wide input IDs must be unique")
    start = 1 + int(metadata_columns)
    value_count = table.shape[1] - start
    if value_count <= 0 or value_count % int(day_points):
        raise ValueError("Dense-wide values must contain complete day blocks")
    if int(day_points) % PHASES:
        raise ValueError("day_points must be divisible by 96")
    days = value_count // int(day_points)
    points_per_phase = int(day_points) // PHASES
    raw = table.iloc[:, start:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    raw[raw <= 0] = np.nan
    raw = _clip_glucose(raw)
    dense = raw.reshape(len(table), days, PHASES, points_per_phase)
    values = dense[..., 0]
    included, excluded, elapsed = _aggregate_fixed_values(
        table.iloc[:, 0].astype(str), values, num_workers, progress_every
    )
    return write_agp_dataset(
        output_dir, included, excluded, {
            "format": "dense_wide_day_blocks",
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "day_blocks": days,
            "day_points": int(day_points),
            "points_per_phase": points_per_phase,
            "phase_selection": "first_point",
            "num_workers": max(1, int(num_workers)),
            "elapsed_seconds": elapsed,
        },
    )



def _process_uva_path(item):
    path, glucose_divisor, sigma_minutes, cutoff_minutes = item
    path = Path(path)
    origin = datetime(2000, 1, 1)
    try:
        table = pd.read_csv(path)
        if "time_min" not in table or "glucose" not in table:
            raise ValueError("UVA input requires time_min and glucose columns")
        minutes = pd.to_numeric(table["time_min"], errors="coerce")
        glucose = pd.to_numeric(table["glucose"], errors="coerce")
        records = [
            (
                origin + timedelta(minutes=float(minute)),
                float(value) / float(glucose_divisor),
            )
            for minute, value in zip(minutes, glucose)
            if np.isfinite(minute) and np.isfinite(value) and value > 0
        ]
        return path.stem, aggregate_subject(
            records, sigma_minutes, cutoff_minutes
        ), None
    except (OSError, ValueError) as error:
        return path.stem, None, str(error)


def build_from_uva_directory(
    input_dir, output_dir, glucose_divisor=18.0,
    num_workers=1, progress_every=1000,
    sigma_minutes=SIGMA_MINUTES, cutoff_minutes=CUTOFF_MINUTES,
):
    """Process one sampling-rate directory of UVA simulator CSV files."""
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("*.csv"))
    workers = max(1, int(num_workers))
    every = max(1, int(progress_every))
    started = wall_time.monotonic()
    items = [
        (path, float(glucose_divisor), float(sigma_minutes), float(cutoff_minutes))
        for path in paths
    ]
    executor = None
    if workers == 1:
        results = map(_process_uva_path, items)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        results = executor.map(_process_uva_path, items)
    included, excluded = [], []
    try:
        for completed, (sample_id, result, error) in enumerate(results, start=1):
            if error is None:
                included.append((sample_id, result))
            else:
                excluded.append({"ID": sample_id, "reason": error})
            if completed == 1 or completed % every == 0 or completed == len(items):
                elapsed = wall_time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(items) - completed) / max(rate, 1e-9)
                print(
                    f"[preprocess] {completed}/{len(items)} "
                    f"({100.0 * completed / max(len(items), 1):.2f}%) "
                    f"rate={rate:.2f} samples/s "
                    f"elapsed={_duration_text(elapsed)} "
                    f"eta={_duration_text(remaining)}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return write_agp_dataset(
        output_dir,
        included,
        excluded,
        {
            "format": "uva_relative_minutes_directory",
            "path": str(input_dir.resolve()),
            "files": len(paths),
            "glucose_input_unit": "mg/dL",
            "glucose_output_unit": "mmol/L",
            "glucose_divisor": float(glucose_divisor),
            "num_workers": workers,
            "elapsed_seconds": wall_time.monotonic() - started,
            "gaussian_sigma_minutes": float(sigma_minutes),
            "open_alignment_cutoff_minutes": float(cutoff_minutes),
        },
    )
