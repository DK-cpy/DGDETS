"""NAS-Bench-101 adapter retaining CDENAG's original fitness call chain."""

from __future__ import absolute_import, division, print_function

import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parent
VENDORED_NASBENCH = CODE_ROOT / "nasbench"
if str(VENDORED_NASBENCH) not in sys.path:
    sys.path.insert(0, str(VENDORED_NASBENCH))

from gdets.spaces import NASBench101Space
from utils.nb101_api import get_nb101_api
from utils.nb101_fitness import arch_fitness


class NASBench101Adapter(object):
    benchmark = "NAS-Bench-101"
    backend = "CDENAG utils.nb101_fitness.arch_fitness (TFRecord query-backed)"
    query_backed = True
    search_metric_name = "NAS-Bench-101 test accuracy"
    search_metric_is_benchmark = True
    source_functions = (
        "utils.nb101_fitness._neural_predictor",
        "utils.nb101_fitness.arch_fitness",
    )

    def __init__(self, asset_root=None, api=None):
        self.asset_root = Path(asset_root or Path(__file__).resolve().parent).resolve()
        self.space = NASBench101Space()
        record = self.asset_root / "nasbench" / "nasbench_only108.tfrecord"
        if api is None:
            if not record.is_file():
                raise FileNotFoundError(
                    "NAS-Bench-101 TFRecord not found: %s. Run prepare_assets.py." % record
                )
            api = get_nb101_api(str(record))
        self.api = api
        self.context = {"dataset": "cifar10", "tfrecord": str(record)}

    def _legacy_tensor(self, genes):
        matrices = [self.space.matrix(self.space.repair(gene)) for gene in genes]
        return torch.tensor(np.asarray(matrices), dtype=torch.float32)

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
