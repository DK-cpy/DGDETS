"""NAS-Bench-301 adapter retaining the supplied zero-cost table interface."""

from __future__ import absolute_import, division, print_function

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from gdets.spaces import NASBench301Space
from utils.NB301 import get_api
from utils.fitness import arch_fitness


@contextmanager
def _working_directory(path):
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


class NASBench301Adapter(object):
    benchmark = "NAS-Bench-301"
    backend = "CDENAG utils.fitness.arch_fitness (zc_nasbench301 lookup)"
    query_backed = True
    search_metric_name = "NAS-Bench-301 table accuracy"
    search_metric_is_benchmark = True
    source_functions = (
        "utils.NB301.get_api",
        "utils.fitness.neural_predictor",
        "utils.fitness.arch_fitness",
    )

    def __init__(self, asset_root=None, api=None):
        self.asset_root = Path(asset_root or Path(__file__).resolve().parent).resolve()
        table = self.asset_root / "zc_nasbench301.json"
        if api is None:
            if not table.is_file():
                raise FileNotFoundError(
                    "NAS-Bench-301 table not found: %s. Run prepare_assets.py." % table
                )
            with _working_directory(self.asset_root):
                api = get_api(verbose=False)
        self.api = api
        self.space = NASBench301Space(self.api.keys())
        self.context = {"dataset": "cifar10", "table": str(table)}

    def _legacy_tensor(self, genes):
        matrix = np.zeros((len(genes), 4, 14), dtype=np.float32)
        for row, gene in enumerate(genes):
            for variable, label in enumerate(self.space.repair(gene)):
                matrix[row, variable, int(label)] = 1.0
        return torch.tensor(matrix, dtype=torch.float32)

    def score_batch(self, genes):
        original_accuracy, original_fitness = arch_fitness(
            adj_matrix=self._legacy_tensor(genes), nb_api=self.api
        )
        return (
            original_accuracy.detach().cpu().numpy(),
            original_fitness.detach().cpu().numpy(),
        )

    def legacy_population(self, genes):
        return self._legacy_tensor(genes).reshape(len(genes), -1)
