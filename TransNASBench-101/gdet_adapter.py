"""TransNASBench-101 adapter retaining CDENAG's supplied API fitness."""

from __future__ import absolute_import, division, print_function

from pathlib import Path

import numpy as np
import torch

from gdets.spaces import TransNASBench101Space
from TransNASBench101.api import TransNASBenchAPI
from utils.transnasbench101_fitness import arch_fitness


class TransNASBench101Adapter(object):
    benchmark = "TransNASBench-101"
    backend = "CDENAG utils.transnasbench101_fitness.arch_fitness (API query-backed)"
    query_backed = True
    search_metric_is_benchmark = True
    source_functions = (
        "utils.transnasbench101_fitness.macro_neural_predictor",
        "utils.transnasbench101_fitness.micro_neural_predictor",
        "utils.transnasbench101_fitness.arch_fitness",
    )

    def __init__(
        self,
        asset_root=None,
        api=None,
        task="class_object",
        search_space="micro",
        avoid_useless=True,
    ):
        self.asset_root = Path(asset_root or Path(__file__).resolve().parent).resolve()
        benchmark_file = (
            self.asset_root / "TransNASBench101" / "transnas-bench_v10141024.pth"
        )
        if api is None:
            if not benchmark_file.is_file():
                raise FileNotFoundError(
                    "TransNASBench-101 file not found: %s. Run prepare_assets.py."
                    % benchmark_file
                )
            api = TransNASBenchAPI(str(benchmark_file))
        if task not in api.task_list:
            raise ValueError("unknown TransNAS task: %s" % task)
        if search_space not in api.search_spaces:
            raise ValueError("unknown TransNAS search space: %s" % search_space)
        self.api = api
        self.task = task
        self.search_space = search_space
        metric_names = {
            "class_scene": "TransNASBench-101 validation top-1 accuracy",
            "class_object": "TransNASBench-101 validation top-1 accuracy",
            "room_layout": "TransNASBench-101 training loss x100 (lower is better)",
            "jigsaw": "TransNASBench-101 validation top-1 accuracy",
            "segmentsemantic": "TransNASBench-101 validation mIoU",
            "normal": "TransNASBench-101 validation SSIM x100",
            "autoencoder": "TransNASBench-101 validation SSIM x100",
        }
        self.search_metric_name = metric_names[task]
        self.metric_higher_is_better = task != "room_layout"
        self.space = TransNASBench101Space(
            api,
            search_space=search_space,
            avoid_useless=avoid_useless,
        )
        self.context = {
            "task": task,
            "search_space": search_space,
            "avoid_useless": bool(avoid_useless),
            "benchmark_file": str(benchmark_file),
        }

    def _legacy_tensor(self, genes):
        cardinality = 5 if self.search_space == "macro" else 4
        matrix = np.zeros((len(genes), 6, cardinality), dtype=np.float32)
        for row, gene in enumerate(genes):
            for variable, label in enumerate(self.space.repair(gene)):
                matrix[row, variable, int(label)] = 1.0
        return torch.tensor(matrix, dtype=torch.float32)

    def score_batch(self, genes):
        original_accuracy, original_fitness = arch_fitness(
            operation_matrix=self._legacy_tensor(genes),
            api=self.api,
            task=self.task,
            search_space=self.search_space,
        )
        return (
            original_accuracy.detach().cpu().numpy(),
            original_fitness.detach().cpu().numpy(),
        )

    def legacy_population(self, genes):
        return self._legacy_tensor(genes).reshape(len(genes), -1)
