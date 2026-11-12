#!/usr/bin/env python3
"""Explicit hover time ablation on the existing production trainer."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffphys.hover_time_experiment import HoverTimeConfig, experiment
from tools import train_response_control


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--hover-time-weight", type=float, required=True)
    extra, remaining = parser.parse_known_args(argv)
    config = HoverTimeConfig(weight=extra.hover_time_weight)
    with experiment(config):
        train_response_control.main(remaining)


if __name__ == "__main__":
    main()
