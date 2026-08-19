"""Run every TransNASBench-101 task/space with the GDETS search engine."""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdets.cli import add_search_arguments, config_from_args
from gdets.experiments import (
    _resolve_seeds,
    _run_record,
    add_experiment_arguments,
    set_random_seed,
)
from gdets.presets import apply_search_profile
from TransNASBench101.api import TransNASBenchAPI
from config.config import experiment_random_seeds, hyper_params_setting
from gdet_adapter import TransNASBench101Adapter


TASKS = (
    "class_scene",
    "class_object",
    "room_layout",
    "jigsaw",
    "segmentsemantic",
    "normal",
    "autoencoder",
)
TASK_ABBREV = {
    "class_object": "OC",
    "class_scene": "SC",
    "autoencoder": "AE",
    "normal": "SE",
    "segmentsemantic": "SS",
    "room_layout": "RP",
    "jigsaw": "JS",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--spaces", nargs="+", choices=("macro", "micro"), default=("macro", "micro")
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--allow-useless", action="store_true")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for the CDENAG-style experiment_results_*.txt file",
    )
    parser = add_experiment_arguments(add_search_arguments(parser))
    parser.set_defaults(
        exp_type="fixed",
        output=Path("results/transnas_all.json"),
        profile="balanced",
    )
    return parser


def _summary(records):
    metrics = [item["benchmark_metric"] for item in records]
    durations = [item["duration_seconds"] for item in records]
    uniqueness = [item["final_population_uniqueness_rate"] for item in records]
    return {
        "number_of_runs": len(records),
        "average_selected_metric": float(np.mean(metrics)),
        "std_selected_metric": (
            float(np.std(metrics, ddof=1)) if len(metrics) > 1 else 0.0
        ),
        "best_selected_metric": (
            float(min(metrics))
            if records and "lower is better" in records[0]["benchmark_metric_name"]
            else float(max(metrics))
        ),
        "average_duration_seconds": float(np.mean(durations)),
        "average_final_population_uniqueness_rate": float(np.mean(uniqueness)),
        "metric_name": records[0]["benchmark_metric_name"],
    }


def main(argv=None):
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args(argv)
    apply_search_profile(args, "transnasbench101", argv=argv)
    asset_root = args.asset_root.expanduser().resolve()
    benchmark_file = asset_root / "TransNASBench101" / "transnas-bench_v10141024.pth"
    if not benchmark_file.is_file():
        raise FileNotFoundError("TransNASBench-101 file not found: %s" % benchmark_file)
    api = TransNASBenchAPI(str(benchmark_file))
    unknown = sorted(set(args.tasks) - set(api.task_list))
    if unknown:
        raise ValueError("tasks absent from the API: %s" % unknown)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.result_dir = args.result_dir.expanduser().resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    text_path = args.result_dir / ("experiment_results_%s.txt" % timestamp)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": "GDETS-CDENAG-Proxy",
        "benchmark": "TransNASBench-101",
        "protocol": {
            "all_requested_tasks": list(args.tasks),
            "all_requested_spaces": list(args.spaces),
            "seed_mode": args.exp_type,
            "config_driven": True,
            "historical_teacher": False,
            "sats": False,
        },
        "search_profile": args.profile,
        "config": asdict(config_from_args(args)),
        "experiments": [],
        "failures": [],
        "text_result_file": str(text_path),
    }

    started = time.time()
    with text_path.open("w", encoding="utf-8") as text_file:
        def emit(message=""):
            print(message)
            text_file.write(str(message) + "\n")
            text_file.flush()

        emit("=" * 80)
        emit("TransNAS-Bench-101 GDETS experiment record | %s" % datetime.now())
        emit("Search engine: few-step categorical graph diffusion + tabu search")
        emit("Retained evaluator: CDENAG TransNASBench-101 API fitness interface")
        emit("Historical teacher / EMA / SATS: disabled")
        emit("Search profile: %s; seed mode: %s" % (args.profile, args.exp_type))
        emit("=" * 80)

        for space in args.spaces:
            emit("\n" + "=" * 55)
            emit(" Start %s search space - %d tasks" % (space.upper(), len(args.tasks)))
            emit("=" * 55)
            for task in args.tasks:
                label = "%s/%s" % (space, task)
                try:
                    settings = dict(hyper_params_setting[space][task])
                    settings["random_seed"] = list(experiment_random_seeds)
                    if args.exp_type == "single":
                        mode, seeds = "single", [int(args.seed)]
                    else:
                        mode, seeds = _resolve_seeds(settings, args.exp_type, args.runs)
                    emit("\n>>> Running experiment with dataset: %s" % label)
                    emit(">>> Seed source: config.py (%s seed list)" % mode)
                    adapter = TransNASBench101Adapter(
                        asset_root,
                        api=api,
                        task=task,
                        search_space=space,
                        avoid_useless=not args.allow_useless,
                    )
                    records = []
                    for index, seed in enumerate(seeds):
                        set_random_seed(seed)
                        run_args = copy.copy(args)
                        run_args.seed = int(seed)
                        emit(
                            "\n>>> Exp %d: Running on %s dataset with seed %d..."
                            % (index, label, seed)
                        )
                        record = _run_record(adapter, run_args)
                        records.append(record)
                        emit(
                            ">>> Search/final metric (%s): %.8f"
                            % (record["benchmark_metric_name"], record["benchmark_metric"])
                        )
                        emit(
                            ">>> Duration: %.2f seconds; final uniqueness rate: %.2f"
                            % (
                                record["duration_seconds"],
                                record["final_population_uniqueness_rate"],
                            )
                        )
                        emit(
                            ">>> Task %s (%s) with seed %d in search space %s, "
                            "selected task metric: %.8f"
                            % (
                                task,
                                TASK_ABBREV[task],
                                seed,
                                space,
                                record["benchmark_metric"],
                            )
                        )
                    summary = _summary(records)
                    emit(
                        "\n>>> In %d %s seed experiment on %s, average selected "
                        "metric is %.8f +/- %.8f, average duration is %.2f seconds, "
                        "average uniqueness rate is %.2f."
                        % (
                            len(records),
                            mode,
                            label,
                            summary["average_selected_metric"],
                            summary["std_selected_metric"],
                            summary["average_duration_seconds"],
                            summary["average_final_population_uniqueness_rate"],
                        )
                    )
                    payload["experiments"].append(
                        {
                            "search_space": space,
                            "task": task,
                            "task_abbreviation": TASK_ABBREV[task],
                            "seed_mode": mode,
                            "seeds": seeds,
                            "summary": summary,
                            "runs": records,
                        }
                    )
                except Exception as error:
                    failure = {"search_space": space, "task": task, "error": repr(error)}
                    payload["failures"].append(failure)
                    emit("\n>>> FAILED %s: %r" % (label, error))

        payload["elapsed_seconds"] = float(time.time() - started)
        emit("\n" + "=" * 80)
        emit("Final result summary")
        emit("=" * 80)
        for experiment in payload["experiments"]:
            item = experiment["summary"]
            emit(
                ">>> Task %s (%s) in search space %s, mean selected metric: "
                "%.8f +/- %.8f; best: %.8f; metric: %s"
                % (
                    experiment["task"],
                    experiment["task_abbreviation"],
                    experiment["search_space"],
                    item["average_selected_metric"],
                    item["std_selected_metric"],
                    item["best_selected_metric"],
                    item["metric_name"],
                )
            )
        emit("Failures: %d" % len(payload["failures"]))
        emit("Total elapsed time: %.2f seconds" % payload["elapsed_seconds"])
        emit("JSON result: %s" % args.output)
        emit("=" * 80)

    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("All-task text result written to %s" % text_path)
    print("All-task JSON result written to %s" % args.output)
    return payload


if __name__ == "__main__":
    main()
