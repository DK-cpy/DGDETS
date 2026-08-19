"""NAS-Bench-201 few-step graph diffusion evolution with original proxy."""

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
from config.config import (
    experiment_random_seeds,
    meta_hyper_params_setting,
    nb201_hyper_params_setting,
)
from gdet_adapter import BENCHMARK_DATASETS, META_DATASETS, NASBench201Adapter


def resolve_proxy_backend(dataset, requested="auto"):
    """Reproduce CDENAG's dataset-specific evaluator routing."""
    expected = "benchmark" if dataset in BENCHMARK_DATASETS else "meta"
    if requested == "auto":
        return expected
    if requested != expected:
        raise ValueError(
            "dataset %s must use --proxy-backend %s (received %s)"
            % (dataset, expected, requested)
        )
    return requested


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--dataset",
        choices=("cifar10", "cifar100", "ImageNet16-120", "aircraft", "pets"),
        default="cifar10",
    )
    parser.add_argument(
        "--proxy-backend",
        choices=("auto", "meta", "benchmark"),
        default="auto",
        help=(
            "auto routes CIFAR-10/CIFAR-100/ImageNet16-120 to the benchmark "
            "table and Aircraft/Pets to MetaD2A"
        ),
    )
    parser.add_argument("--api-path", type=Path)
    parser.add_argument(
        "--meta-ensemble", type=int, default=1,
        help="number of deterministic coreset views used by the retained MetaD2A proxy",
    )
    parser.add_argument(
        "--meta-uncertainty-beta", type=float, default=0.0,
        help="lower-confidence-bound penalty applied only to search utility",
    )
    parser.add_argument(
        "--meta-ensemble-seed", type=int,
        help=(
            "optional fixed coreset seed; by default each experiment uses its "
            "config.py seed exactly as CDENAG"
        ),
    )
    parser.add_argument(
        "--share-meta-encoding",
        action="store_true",
        help="reuse one MetaD2A dataset encoding across seeds (faster, not CDENAG-equivalent)",
    )
    parser.add_argument(
        "--meta-final-eval",
        choices=("train", "proxy"),
        default="train",
        help=(
            "train reproduces CDENAG's downstream top-k training and reports real "
            "Aircraft/Pets Top-1; proxy is a fast search-only diagnostic"
        ),
    )
    parser.add_argument(
        "--meta-final-topk",
        type=int,
        help="override config.py top-k for Aircraft/Pets downstream training",
    )
    parser.add_argument(
        "--validity", choices=("benchmark", "effective", "strict"), default="strict"
    )
    return add_experiment_arguments(add_search_arguments(parser))


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_search_profile(args, "nasbench201", argv=argv)
    args.proxy_backend = resolve_proxy_backend(args.dataset, args.proxy_backend)
    explicit_argv = list(sys.argv[1:] if argv is None else argv)
    if (
        args.proxy_backend == "benchmark"
        and "--consistency-rerank-top-k" not in explicit_argv
    ):
        # Neighborhood consistency is a MetaD2A anti-spike calibration, not a
        # transformation of exact benchmark labels.
        args.consistency_rerank_top_k = 0
        args.consistency_weight = 0.0
    if args.meta_ensemble < 1:
        raise ValueError("--meta-ensemble must be positive")
    if args.meta_uncertainty_beta < 0:
        raise ValueError("--meta-uncertainty-beta cannot be negative")
    settings_source = (
        nb201_hyper_params_setting
        if args.proxy_backend == "benchmark"
        else meta_hyper_params_setting
    )
    settings = dict(settings_source[args.dataset])
    settings["random_seed"] = list(experiment_random_seeds)

    cached_api = [None]
    cached_meta = [None]

    def adapter_factory(seed):
        adapter = NASBench201Adapter(
            args.asset_root,
            dataset=args.dataset,
            proxy_backend=args.proxy_backend,
            api=cached_api[0],
            api_path=args.api_path,
            validity=args.validity,
            seed=(
                int(args.meta_ensemble_seed)
                if args.meta_ensemble_seed is not None
                else int(seed)
            ),
            meta_ensemble=args.meta_ensemble,
            meta_uncertainty_beta=args.meta_uncertainty_beta,
            shared_meta=(cached_meta[0] if args.share_meta_encoding else None),
            meta_final_eval=args.meta_final_eval,
            meta_final_topk=args.meta_final_topk,
            meta_eval_settings=settings,
        )
        if (
            args.proxy_backend == "meta"
            and args.share_meta_encoding
            and cached_meta[0] is None
        ):
            cached_meta[0] = adapter.shared_meta_state()
        if adapter.final_api is not None:
            cached_api[0] = adapter.final_api
        return adapter

    return run_config_experiment(adapter_factory, args, settings, args.dataset)


if __name__ == "__main__":
    main()
