"""Command-line entry for GlucoScope V1 pretraining."""

from __future__ import annotations

import argparse

from train.pretrain_engine import run_pretraining
from utils.config import apply_overrides, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/pretrain.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted YAML override, e.g. train.epochs=1",
    )
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    return run_pretraining(config)


if __name__ == "__main__":
    main()
