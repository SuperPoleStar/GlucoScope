"""Build GlucoScope 96 x 7 AGP inputs from raw CGM data."""

from __future__ import annotations

import argparse
import json

from provider.cgm_preprocessing import (
    build_from_dense_wide,
    build_from_directories,
    build_from_directory,
    build_from_uva_directory,
    build_from_wide,
)
from utils.config import apply_overrides, load_config, project_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/preprocess.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    output = project_path(config["output"]["directory"])
    if config["input"]["format"] == "directory":
        source = project_path(config["input"]["path"])
        manifest = build_from_directory(
            source,
            output,
            config["input"].get("gaussian_sigma_minutes", 5.0),
            config["input"].get("gaussian_cutoff_minutes", 15.0),
        )
    elif config["input"]["format"] == "directories":
        sources = [project_path(value) for value in config["input"]["paths"]]
        manifest = build_from_directories(
            sources,
            output,
            config["input"].get("max_files_per_directory"),
            config["input"].get("sample_seed", 42),
            config["input"].get("num_workers", 1),
            config["input"].get("progress_every", 1000),
            config["input"].get("gaussian_sigma_minutes", 5.0),
            config["input"].get("gaussian_cutoff_minutes", 15.0),
        )
    elif config["input"]["format"] == "wide":
        source = project_path(config["input"]["path"])
        manifest = build_from_wide(
            source,
            output,
            config["input"].get("block_width", 98),
            config["input"].get("num_workers", 1),
            config["input"].get("progress_every", 1000),
        )
    elif config["input"]["format"] == "dense_wide":
        source = project_path(config["input"]["path"])
        manifest = build_from_dense_wide(
            source,
            output,
            config["input"].get("day_points", 288),
            config["input"].get("metadata_columns", 1),
            config["input"].get("num_workers", 1),
            config["input"].get("progress_every", 1000),
        )
    elif config["input"]["format"] == "uva_directory":
        source = project_path(config["input"]["path"])
        manifest = build_from_uva_directory(
            source,
            output,
            config["input"].get("glucose_divisor", 18.0),
            config["input"].get("num_workers", 1),
            config["input"].get("progress_every", 1000),
            config["input"].get("gaussian_sigma_minutes", 5.0),
            config["input"].get("gaussian_cutoff_minutes", 15.0),
        )
    else:
        raise ValueError(
            "input.format must be directory, directories, wide, dense_wide, or uva_directory"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
