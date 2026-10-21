"""Strict 96 x 7 AGP table and effective-day metadata loader."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


FEATURE_NAMES = ("max", "p95", "p75", "median", "p25", "p5", "min")


def parse_agp_cell(value):
    text = str(value).strip()
    if not text or text == "{}":
        raise ValueError("GlucoScope requires all seven quantiles at every phase")
    values = ast.literal_eval("[" + text.strip("{}") + "]")
    if len(values) != 7:
        raise ValueError(f"Expected seven quantiles, got {len(values)}")
    result = [float(item) for item in values]
    if not np.isfinite(result).all():
        raise ValueError("AGP phase contains non-finite values")
    return result


def validate_quantile_order(data, tolerance=1e-5):
    """Return fraction of phases violating max >= ... >= min."""
    differences = data[..., :-1] - data[..., 1:]
    return float((differences < -float(tolerance)).any(axis=-1).mean())


class GlucoScopePretrainDataset(Dataset):
    """Load paired AGP and remark CSV files without fitting normalization."""

    def __init__(
        self,
        data_path,
        remark_path,
        day_count_column="effective_days",
        min_day_count=1.0,
        strict_quantile_order=False,
    ):
        self.data_path = Path(data_path)
        self.remark_path = Path(remark_path)
        if not self.data_path.is_file() or not self.remark_path.is_file():
            raise FileNotFoundError("Both data_path and remark_path must exist")
        self.ids, self.data = self._load_agp(self.data_path)
        metadata = self._load_remark(self.remark_path, day_count_column)
        missing = [sample_id for sample_id in self.ids if sample_id not in metadata]
        if missing:
            raise ValueError(f"Remark file misses {len(missing)} AGP IDs")
        self.raw_day_count = np.asarray(
            [metadata[item][0] for item in self.ids], dtype=np.float32
        )
        self.phase_effective_days = np.stack(
            [metadata[item][1] for item in self.ids]
        ).astype(np.float32)
        self.monitoring_days = np.asarray(
            [metadata[item][2] for item in self.ids], dtype=np.float32
        )
        keep = self.raw_day_count >= float(min_day_count)
        self.ids = [item for item, selected in zip(self.ids, keep) if selected]
        self.data = self.data[keep]
        self.raw_day_count = self.raw_day_count[keep]
        self.phase_effective_days = self.phase_effective_days[keep]
        self.monitoring_days = self.monitoring_days[keep]
        if not self.ids:
            raise ValueError("No samples remain after min_day_count filtering")
        self.quantile_order_violation_rate = validate_quantile_order(self.data)
        if strict_quantile_order and self.quantile_order_violation_rate > 0:
            raise ValueError("AGP quantile ordering violations detected")

    @staticmethod
    def _load_agp(path):
        ids, samples = [], []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if len(header) != 97 or header[0] != "ID":
                raise ValueError("AGP CSV must contain ID plus 96 phase columns")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != 97:
                    raise ValueError(f"Invalid row length at line {line_number}")
                ids.append(str(row[0]))
                samples.append([parse_agp_cell(cell) for cell in row[1:]])
        if len(ids) != len(set(ids)):
            raise ValueError("AGP IDs must be unique")
        data = np.asarray(samples, dtype=np.float32)
        if data.shape[1:] != (96, 7):
            raise ValueError(f"Expected (N,96,7), got {data.shape}")
        return ids, data

    @staticmethod
    def _load_remark(path, day_count_column):
        phase_columns = [f"phase_{index}_effective_days" for index in range(96)]
        result = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"ID", day_count_column, *phase_columns}
            if not required.issubset(reader.fieldnames or []):
                missing = sorted(required - set(reader.fieldnames or []))
                raise ValueError(f"Remark CSV lacks columns: {missing[:5]}")
            for row in reader:
                sample_id = str(row["ID"])
                phases = np.asarray(
                    [float(row[column]) for column in phase_columns],
                    dtype=np.float32,
                )
                monitoring_days = float(
                    row.get("monitoring_days") or row[day_count_column]
                )
                sample_days = float(row[day_count_column])
                if sample_days < 0 or np.any(phases < 0):
                    raise ValueError(f"Negative effective days for ID {sample_id}")
                if not np.isclose(sample_days, float(phases.min())):
                    raise ValueError(
                        f"{day_count_column} must equal min phase days for ID {sample_id}"
                    )
                result[sample_id] = (sample_days, phases, monitoring_days)
        return result

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {
            "id": self.ids[index],
            "x": torch.from_numpy(self.data[index].copy()),
            "raw_day_count": torch.tensor(self.raw_day_count[index]),
            "monitoring_days": torch.tensor(self.monitoring_days[index]),
            "phase_effective_days": torch.from_numpy(
                self.phase_effective_days[index].copy()
            ),
        }

    def audit(self):
        return {
            "samples": len(self),
            "shape": list(self.data.shape),
            "glucose_min": float(self.data.min()),
            "glucose_max": float(self.data.max()),
            "missing_values": int(np.isnan(self.data).sum()),
            "quantile_order_violation_rate": self.quantile_order_violation_rate,
            "effective_days_min": float(self.raw_day_count.min()),
            "effective_days_median": float(np.median(self.raw_day_count)),
            "effective_days_max": float(self.raw_day_count.max()),
        }


def split_pretraining_dataset(dataset, validation_ratio=0.1, seed=42):
    """Deterministic ID-level split; validation is never used for calibration."""
    if not 0 < float(validation_ratio) < 1:
        raise ValueError("validation_ratio must be in (0,1)")
    indices = np.arange(len(dataset))
    np.random.default_rng(seed).shuffle(indices)
    validation_size = max(1, int(round(len(indices) * validation_ratio)))
    validation = indices[:validation_size].tolist()
    training = indices[validation_size:].tolist()
    if not training:
        raise ValueError("Training split is empty")
    return Subset(dataset, training), Subset(dataset, validation)



class GlucoScopeMaterializedPretrainDataset(Dataset):
    """Load a materialized pretraining bundle without parsing AGP CSV cells."""

    REQUIRED_ARRAYS = {
        "ids", "data", "effective_days", "monitoring_days", "phase_effective_days"
    }

    def __init__(self, path, min_day_count=1.0, strict_quantile_order=False):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Materialized pretraining dataset not found: {self.path}")
        with np.load(self.path, allow_pickle=False) as bundle:
            missing = self.REQUIRED_ARRAYS - set(bundle.files)
            if missing:
                raise ValueError(f"Materialized pretraining dataset lacks arrays: {sorted(missing)}")
            self.ids = [str(value) for value in bundle["ids"].tolist()]
            self.data = np.asarray(bundle["data"], dtype=np.float32)
            self.raw_day_count = np.asarray(bundle["effective_days"], dtype=np.float32)
            self.monitoring_days = np.asarray(bundle["monitoring_days"], dtype=np.float32)
            self.phase_effective_days = np.asarray(
                bundle["phase_effective_days"], dtype=np.float32
            )
        samples = len(self.ids)
        actual = {
            "data": self.data.shape,
            "effective_days": self.raw_day_count.shape,
            "monitoring_days": self.monitoring_days.shape,
            "phase_effective_days": self.phase_effective_days.shape,
        }
        expected = {
            "data": (samples, 96, 7),
            "effective_days": (samples,),
            "monitoring_days": (samples,),
            "phase_effective_days": (samples, 96),
        }
        if actual != expected:
            raise ValueError(f"Materialized pretraining array shapes invalid: {actual}")
        if samples != len(set(self.ids)):
            raise ValueError("Materialized pretraining IDs must be unique")
        if not np.isfinite(self.data).all():
            raise ValueError("Materialized pretraining data contains non-finite values")
        if np.any(self.raw_day_count < 0) or np.any(self.phase_effective_days < 0):
            raise ValueError("Materialized pretraining effective days must be nonnegative")
        if not np.allclose(self.raw_day_count, self.phase_effective_days.min(axis=1)):
            raise ValueError("effective_days must equal minimum phase effective days")
        keep = self.raw_day_count >= float(min_day_count)
        self.ids = [item for item, selected in zip(self.ids, keep) if selected]
        self.data = self.data[keep]
        self.raw_day_count = self.raw_day_count[keep]
        self.monitoring_days = self.monitoring_days[keep]
        self.phase_effective_days = self.phase_effective_days[keep]
        if not self.ids:
            raise ValueError("No samples remain after min_day_count filtering")
        self.quantile_order_violation_rate = validate_quantile_order(self.data)
        if strict_quantile_order and self.quantile_order_violation_rate > 0:
            raise ValueError("AGP quantile ordering violations detected")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {
            "id": self.ids[index],
            "x": torch.from_numpy(self.data[index].copy()),
            "raw_day_count": torch.tensor(self.raw_day_count[index]),
            "monitoring_days": torch.tensor(self.monitoring_days[index]),
            "phase_effective_days": torch.from_numpy(
                self.phase_effective_days[index].copy()
            ),
        }

    def audit(self):
        return {
            "samples": len(self),
            "shape": list(self.data.shape),
            "materialized_path": str(self.path),
            "glucose_min": float(self.data.min()),
            "glucose_max": float(self.data.max()),
            "missing_values": int(np.isnan(self.data).sum()),
            "quantile_order_violation_rate": self.quantile_order_violation_rate,
            "effective_days_min": float(self.raw_day_count.min()),
            "effective_days_median": float(np.median(self.raw_day_count)),
            "effective_days_max": float(self.raw_day_count.max()),
        }
