"""TransNASBench-101 few-step graph diffusion evolution with tabu search."""

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
from gdet_adapter import TransNASBench101Adapter


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--task",
        choices=(
            "class_scene", "class_object", "room_layout", "jigsaw",
            "segmentsemantic", "normal", "autoencoder",
        ),
        default="class_object",
    )
    parser.add_argument("--search-space", choices=("macro", "micro"), default="micro")
    parser.add_argument("--allow-useless", action="store_true")
    return add_experiment_arguments(add_search_arguments(parser))


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_search_profile(args, "transnasbench101", argv=argv)
    settings = dict(hyper_params_setting[args.search_space][args.task])
    settings["random_seed"] = list(experiment_random_seeds)
    cached_api = [None]

    def adapter_factory(seed):
        adapter = TransNASBench101Adapter(
            args.asset_root,
            api=cached_api[0],
            task=args.task,
            search_space=args.search_space,
            avoid_useless=not args.allow_useless,
        )
        cached_api[0] = adapter.api
        return adapter

    label = "%s/%s" % (args.search_space, args.task)
    return run_config_experiment(adapter_factory, args, settings, label)


if __name__ == "__main__":
    main()
