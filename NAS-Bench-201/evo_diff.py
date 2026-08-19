"""Compatibility entries for the original NAS-Bench-201 experiments."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdets.legacy import run_legacy
from gdet_adapter import NASBench201Adapter


def evo_diff(dataset, api, num_step, population_num, seed, **kwargs):
    adapter = NASBench201Adapter(
        api=api, dataset=dataset, proxy_backend="benchmark", seed=seed
    )
    result, legacy, unique_rate = run_legacy(
        adapter, num_step=num_step, population_num=population_num, seed=seed, **kwargs
    )
    return result.best_reported_metric, result.elapsed_seconds, unique_rate, legacy


def evo_diff_meta(dataset, api, num_step, population_num, seed, **kwargs):
    adapter = NASBench201Adapter(
        dataset=dataset, proxy_backend="meta", seed=seed
    )
    result, legacy, unique_rate = run_legacy(
        adapter, num_step=num_step, population_num=population_num, seed=seed, **kwargs
    )
    return result.best_reported_metric, result.elapsed_seconds, unique_rate, legacy

