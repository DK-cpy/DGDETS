"""NAS-Bench-301 few-step graph diffusion evolution with tabu search."""

from __future__ import absolute_import, division, print_function

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdets.cli import add_search_arguments
from gdets.experiments import add_experiment_arguments, run_config_experiment
from gdets.presets import apply_search_profile
from config.config import experiment_random_seeds, hyper_params_setting
from gdet_adapter import NASBench301Adapter


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parent)
    return add_experiment_arguments(add_search_arguments(parser))


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_search_profile(args, "nasbench301", argv=argv)
    settings = dict(hyper_params_setting)
    settings["random_seed"] = list(experiment_random_seeds)
    cached_api = [None]

    def adapter_factory(seed):
        adapter = NASBench301Adapter(args.asset_root, api=cached_api[0])
        cached_api[0] = adapter.api
        return adapter

    return run_config_experiment(adapter_factory, args, settings, "cifar10")


if __name__ == "__main__":
    main()
