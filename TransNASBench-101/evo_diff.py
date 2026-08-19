"""Compatibility entry for original TransNASBench-101 experiment scripts."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdets.legacy import run_legacy
from gdet_adapter import TransNASBench101Adapter


def evo_diff(task, search_space, api, num_step, population_num, seed, **kwargs):
    adapter = TransNASBench101Adapter(
        api=api, task=task, search_space=search_space, avoid_useless=True
    )
    result, legacy, _ = run_legacy(
        adapter, num_step=num_step, population_num=population_num, seed=seed, **kwargs
    )
    return result.best_reported_metric, result.elapsed_seconds, legacy

