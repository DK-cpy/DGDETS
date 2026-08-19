"""NAS-Bench-201 adapter preserving CDENAG's original MetaD2A proxy."""

from __future__ import absolute_import, division, print_function

from pathlib import Path

import numpy as np
import torch

from gdets.spaces import NASBench201Space
from meta_acc_predictor.unnoised_model import MetaSurrogateUnnoisedModel
from utils.mapping import ReScale
from utils.meta_d2a import MetaTestDataset, meta_neural_predictor
from utils.nb201_fitness import arch_fitness as benchmark_arch_fitness


BENCHMARK_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
META_DATASETS = ("aircraft", "pets")


class _CompactNASBench201API(object):
    """Minimal query interface backed by the supplied 15,625-label table."""

    def __init__(self, data):
        self.data = data
        self._index = {
            architecture: index
            for index, architecture in enumerate(data["arch"]["str"])
        }

    def query_index_by_arch(self, architecture):
        return int(self._index.get(architecture, -1))

    def query_test_acc_by_index(self, index, dataset):
        key = {
            "cifar10": "cifar10",
            "cifar100": "cifar100",
            "ImageNet16-120": "imagenet16-120",
        }.get(dataset)
        if key is None:
            raise ValueError("compact NAS-Bench-201 labels do not contain %s" % dataset)
        if int(index) < 0:
            raise ValueError("architecture is absent from NAS-Bench-201")
        return float(self.data["test-acc"][key][int(index)])


class _InMemoryFitnessRestorer(object):
    def __init__(self, dataset_encoding):
        self.dataset_encoding = dataset_encoding
        self.cache = {}

    def get_coreset_encode(self):
        return self.dataset_encoding

    def get_fitness(self, architecture):
        return self.cache.get(architecture)

    def push_fitness(self, architecture, fitness):
        self.cache[architecture] = float(fitness)


class NASBench201Adapter(object):
    benchmark = "NAS-Bench-201"

    def __init__(
        self,
        asset_root=None,
        dataset="cifar10",
        proxy_backend="benchmark",
        api=None,
        api_path=None,
        validity="strict",
        seed=0,
        meta_ensemble=1,
        meta_uncertainty_beta=0.0,
        shared_meta=None,
        meta_final_eval="train",
        meta_final_topk=None,
        meta_eval_settings=None,
    ):
        if dataset not in ("cifar10", "cifar100", "ImageNet16-120", "aircraft", "pets"):
            raise ValueError("unsupported NAS-Bench-201 dataset: %s" % dataset)
        if proxy_backend not in ("meta", "benchmark"):
            raise ValueError("proxy_backend must be meta or benchmark")
        if proxy_backend == "meta" and dataset not in META_DATASETS:
            raise ValueError(
                "MetaD2A is reserved for transfer datasets %s; %s must use the "
                "NAS-Bench-201 benchmark backend" % (META_DATASETS, dataset)
            )
        if proxy_backend == "benchmark" and dataset not in BENCHMARK_DATASETS:
            raise ValueError(
                "%s has no NAS-Bench-201 benchmark labels; use MetaD2A" % dataset
            )
        if meta_final_eval not in ("train", "proxy"):
            raise ValueError("meta_final_eval must be train or proxy")
        self.asset_root = Path(asset_root or Path(__file__).resolve().parent).resolve()
        self.dataset = dataset
        self.proxy_backend = proxy_backend
        self.space = NASBench201Space(validity=validity)
        self._rescale = ReScale()
        self.meta_ensemble = max(1, int(meta_ensemble))
        self.meta_uncertainty_beta = max(0.0, float(meta_uncertainty_beta))
        self.meta_final_eval = meta_final_eval
        self.meta_final_topk = (
            None if meta_final_topk is None else max(1, int(meta_final_topk))
        )
        self.meta_eval_settings = dict(meta_eval_settings or {})
        self.meta_shared_across_runs = shared_meta is not None
        self.context = {
            "dataset": dataset,
            "proxy_backend": proxy_backend,
            "validity": validity,
            "meta_ensemble": self.meta_ensemble if proxy_backend == "meta" else 0,
            "meta_uncertainty_beta": (
                self.meta_uncertainty_beta if proxy_backend == "meta" else 0.0
            ),
            "meta_final_evaluation": (
                self.meta_final_eval if proxy_backend == "meta" else "not_applicable"
            ),
        }
        self.final_api = None
        self._api_kind = None
        if proxy_backend == "meta":
            self.backend = "CDENAG MetaD2A transferable neural predictor"
            self.query_backed = False
            self.search_metric_name = (
                "MetaD2A calibrated proxy score (ranking only; not trained Top-1)"
            )
            self.search_metric_is_benchmark = False
            self.source_functions = (
                "utils.meta_d2a.meta_neural_predictor",
                "meta_acc_predictor.unnoised_model.MetaSurrogateUnnoisedModel",
                "utils.mapping.ReScale",
            )
            if shared_meta is None:
                self._load_meta_proxy(seed)
            else:
                self._restore_shared_meta(shared_meta)
            if api is not None:
                self.final_api = api
            elif api_path is not None:
                from nas_201_api import NASBench201API

                path = Path(api_path).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError("NAS-Bench-201 final API not found: %s" % path)
                self.final_api = NASBench201API(str(path), verbose=False)
        else:
            self.backend = "CDENAG NAS-Bench-201 benchmark evaluator"
            self.query_backed = True
            self.search_metric_name = "NAS-Bench-201 test accuracy"
            self.search_metric_is_benchmark = True
            self.source_functions = (
                "utils.nb201_fitness.neural_predictor",
                "utils.nb201_fitness.arch_fitness",
            )
            if api is not None:
                if isinstance(api, _CompactNASBench201API):
                    self._api_kind = "compact"
                    self.nasbench201 = api.data
                    self._benchmark_index = dict(api._index)
                else:
                    self._api_kind = "full"
            else:
                full_path = Path(
                    api_path
                    or self.asset_root
                    / "nas_201_api"
                    / "NAS-Bench-201-v1_0-e61699.pth"
                ).resolve()
                if full_path.is_file():
                    from nas_201_api import NASBench201API

                    api = NASBench201API(str(full_path), verbose=False)
                    self._api_kind = "full"
                elif api_path is not None:
                    raise FileNotFoundError("NAS-Bench-201 API not found: %s" % full_path)
                else:
                    compact_path = (
                        self.asset_root / "meta_acc_predictor" / "data" / "nasbench201.pt"
                    )
                    if not compact_path.is_file():
                        raise FileNotFoundError(
                            "neither the full NAS-Bench-201 API nor compact labels exist"
                        )
                    compact_data = torch.load(str(compact_path), map_location="cpu")
                    api = _CompactNASBench201API(compact_data)
                    self.nasbench201 = compact_data
                    self._benchmark_index = dict(api._index)
                    self._api_kind = "compact"
            self.api = api
            self.final_api = api
            if self._api_kind == "compact":
                self.backend += " (compact 15,625-architecture test table)"
                self.source_functions = (
                    "utils.nb201_fitness.arch_fitness",
                    "meta_acc_predictor/data/nasbench201.pt",
                )

    def _load_meta_proxy(self, seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_root = self.asset_root / "meta_acc_predictor" / "data"
        graph_path = data_root / "nasbench201.pt"
        dataset_root = data_root / "meta_predictor_dataset"
        checkpoint = self.asset_root / "meta_acc_predictor" / "unnoised_checkpoint.pth.tar"
        required = [
            graph_path,
            dataset_root / (self.dataset + "bylabel.pt"),
            checkpoint,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "missing CDENAG MetaD2A assets: %s. Run prepare_assets.py."
                % ", ".join(missing)
            )
        self.test_dataset = MetaTestDataset(
            data_path=str(dataset_root), data_name=self.dataset, num_sample=20
        )
        graph_config = {
            "num_vertex_type": 7,
            "max_n": 8,
            "START_TYPE": 0,
            "END_TYPE": 1,
        }
        self.model = MetaSurrogateUnnoisedModel(
            nvt=7, hs=512, nz=56, num_sample=20, graph_config=graph_config
        )
        state = torch.load(str(checkpoint), map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        self.nasbench201 = torch.load(str(graph_path), map_location="cpu")
        self._benchmark_index = {
            architecture: index
            for index, architecture in enumerate(self.nasbench201["arch"]["str"])
        }

        self.restorers = []
        from fast_pytorch_kmeans import KMeans
        for member in range(self.meta_ensemble):
            coreset = []
            for class_index in range(self.test_dataset.num_class):
                images = self.test_dataset.x[class_index][0].to(device)
                # The bundled KMeans fallback initializes from the first item.
                # A deterministic cyclic bootstrap creates distinct, reproducible
                # coreset views without changing or retraining the supplied proxy.
                shift = 0 if member == 0 else (
                    int(seed) + 104729 * member + 7919 * class_index
                ) % max(int(images.shape[0]), 1)
                ordered_images = torch.roll(images, shifts=int(shift), dims=0)
                cluster = KMeans(n_clusters=20, mode="euclidean")
                cluster.fit_predict(ordered_images)
                flat_images = ordered_images.reshape(ordered_images.shape[0], -1).float()
                flat_centers = cluster.centroids.reshape(20, -1).float()
                nearest = torch.argmin(torch.cdist(flat_centers, flat_images), dim=1)
                coreset.append(ordered_images[nearest])
            coreset = torch.cat(coreset, dim=0)
            batch = torch.stack([coreset] * 10).to(device)
            with torch.no_grad():
                dataset_encoding = self.model.set_encode(batch)
            self.restorers.append(_InMemoryFitnessRestorer(dataset_encoding))
        self.device = device

    def shared_meta_state(self):
        if self.proxy_backend != "meta":
            return None
        return {
            "test_dataset": self.test_dataset,
            "model": self.model,
            "nasbench201": self.nasbench201,
            "benchmark_index": self._benchmark_index,
            "restorers": self.restorers,
            "device": self.device,
            "meta_ensemble": self.meta_ensemble,
        }

    def _restore_shared_meta(self, state):
        if int(state["meta_ensemble"]) != self.meta_ensemble:
            raise ValueError("shared MetaD2A state has a different ensemble size")
        self.test_dataset = state["test_dataset"]
        self.model = state["model"]
        self.nasbench201 = state["nasbench201"]
        self._benchmark_index = state["benchmark_index"]
        self.restorers = state["restorers"]
        self.device = state["device"]

    def _legacy_tensor(self, genes):
        matrix = np.full((len(genes), 8, 7), -10.0, dtype=np.float32)
        matrix[:, 0, 0] = 10.0
        matrix[:, 7, 1] = 10.0
        for row, gene in enumerate(genes):
            for edge, operation in enumerate(self.space.repair(gene), start=1):
                matrix[row, edge, int(operation) + 2] = 10.0
        return torch.tensor(matrix, dtype=torch.float32)

    def score_batch(self, genes):
        if self.proxy_backend == "benchmark":
            raw, mapped, _ = benchmark_arch_fitness(
                operation_matrix=self._legacy_tensor(genes),
                api=self.api,
                dataset=self.dataset,
            )
            return raw.detach().cpu().numpy(), mapped.detach().cpu().numpy()

        architecture = [self.space.decode(gene) for gene in genes]
        predictions = []
        for restorer in self.restorers:
            predictions.append(
                np.asarray(
                    meta_neural_predictor(
                        test_dataset=self.test_dataset,
                        meta_surrogate_unnoised_model=self.model,
                        arch_str_list=architecture,
                        dataset_name=self.dataset,
                        nasbench201=self.nasbench201,
                        fitness_restorer=restorer,
                    ),
                    dtype=np.float64,
                )
            )
        predictions = np.stack(predictions, axis=0)
        raw = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        # Preserve the intended dataset calibration in meta_arch_fitness while
        # replacing its identity-comparison bug with deterministic equality.
        if self.dataset == "aircraft":
            raw = raw * 0.92
        elif self.dataset == "pets":
            raw = raw * 1.04
        robust_raw = raw - self.meta_uncertainty_beta * uncertainty
        mapped = self._rescale(torch.tensor(robust_raw, dtype=torch.float32))
        return raw, mapped.detach().cpu().numpy()

    def proxy_metadata(self):
        if self.proxy_backend != "meta":
            return {
                "benchmark_backend_storage": self._api_kind,
                "search_metric_is_benchmark": True,
            }
        return {
            "meta_ensemble_members": int(self.meta_ensemble),
            "meta_uncertainty_beta": float(self.meta_uncertainty_beta),
            "search_utility": "mean_proxy_accuracy - beta * ensemble_std",
            "proxy_shared_across_seed_runs": bool(self.meta_shared_across_runs),
            "reported_search_value_is_not_trained_accuracy": True,
            "final_evaluation_mode": self.meta_final_eval,
        }

    def legacy_population(self, genes):
        return self._legacy_tensor(genes).reshape(len(genes), -1)

    def final_evaluate(self, gene):
        """Evaluate the selected architecture only; never used by the search.

        The compact MetaD2A asset contains the NAS-Bench-201 test accuracies for
        CIFAR-10, CIFAR-100 and ImageNet16-120.  Aircraft and Pets are transfer
        tasks and therefore have no final benchmark label in this file.
        """
        architecture = self.space.decode(gene)
        if self.final_api is not None and self._api_kind == "full":
            index = self.final_api.query_index_by_arch(architecture)
            test_accuracy = float(
                self.final_api.query_test_acc_by_index(index, self.dataset)
            )
            validation_dataset = (
                "cifar10-valid" if self.dataset == "cifar10" else self.dataset
            )
            info = self.final_api.get_more_info(
                index,
                validation_dataset,
                hp="200",
                is_random=False,
            )
            validation_accuracy = info.get("valid-accuracy")
            return {
                "available": True,
                "metric": test_accuracy,
                "metric_name": "NAS-Bench-201 test accuracy",
                "benchmark_validation_accuracy": (
                    None
                    if validation_accuracy is None
                    else float(validation_accuracy)
                ),
                "benchmark_test_accuracy": test_accuracy,
                "evaluation_source": "full NAS-Bench-201 API",
                "additional_queries_after_search": 1,
            }
        key = {
            "cifar10": "cifar10",
            "cifar100": "cifar100",
            "ImageNet16-120": "imagenet16-120",
        }.get(self.dataset)
        if key is None:
            return {
                "available": False,
                "reason": (
                    "MetaD2A output is a ranking score; use --meta-final-eval train "
                    "for CDENAG-style top-k training on %s" % self.dataset
                ),
                "proxy_score_is_not_accuracy": True,
                "additional_queries_after_search": 0,
            }
        index = self._benchmark_index.get(architecture)
        if index is None:
            return {
                "available": False,
                "reason": "selected architecture is absent from compact NB201 labels",
                "additional_queries_after_search": 0,
            }
        test_accuracy = float(self.nasbench201["test-acc"][key][index])
        return {
            "available": True,
            "metric": test_accuracy,
            "metric_name": "NAS-Bench-201 test accuracy",
            "benchmark_validation_accuracy": None,
            "benchmark_test_accuracy": test_accuracy,
            "evaluation_source": "compact MetaD2A test labels (validation unavailable)",
            "additional_queries_after_search": 1,
        }

    def final_evaluate_result(self, result):
        """Keep proxy search and downstream task training strictly separated."""
        if self.proxy_backend != "meta" or self.meta_final_eval != "train":
            return self.final_evaluate(result.best_gene)

        settings = self.meta_eval_settings
        required = (
            "image_cutout", "batch_size", "LR", "momentum", "decay",
            "nesterov", "epochs", "warmup", "eta_min", "multi_thread",
            "early_stop", "topk",
        )
        missing = [key for key in required if key not in settings]
        if missing:
            raise ValueError("missing MetaD2A final-evaluation settings: %s" % missing)
        topk = self.meta_final_topk or int(settings["topk"])
        topk = min(topk, len(result.ranked_genes))
        candidates = self._legacy_tensor(result.ranked_genes[:topk])
        from utils.eval_arch import eval_architectures

        evaluated = eval_architectures(
            x=candidates.cpu(),
            api=_CompactNASBench201API(self.nasbench201),
            dataset_name=self.dataset,
            image_cutout=int(settings["image_cutout"]),
            batch_size=int(settings["batch_size"]),
            device="cuda" if torch.cuda.is_available() else "cpu",
            lr=float(settings["LR"]),
            momentum=float(settings["momentum"]),
            decay=float(settings["decay"]),
            nesterov=bool(settings["nesterov"]),
            train_epochs=int(settings["epochs"]),
            warmup_epoch=int(settings["warmup"]),
            eta_min=float(settings["eta_min"]),
            multi_thread=bool(settings["multi_thread"]),
            early_stop=bool(settings["early_stop"]),
        )
        # The original helper returned only ``max_acc`` in multi-thread mode
        # and ``(max_acc, acc_list)`` in serial mode.  Accept both forms so an
        # experimental config switch cannot break result serialization.
        if isinstance(evaluated, tuple):
            maximum, accuracies = evaluated
        else:
            maximum = evaluated
            accuracies = [maximum]
        return {
            "available": True,
            "metric": float(maximum),
            "metric_name": "%s trained test Top-1 accuracy" % self.dataset,
            "candidate_test_accuracies": [float(value) for value in accuracies],
            "topk_trained": int(topk),
            "evaluation_source": "CDENAG top-k downstream training",
            "additional_queries_after_search": 0,
            "additional_full_trainings_after_search": int(topk),
        }

    def benchmark_summary(self):
        key = {
            "cifar10": "cifar10",
            "cifar100": "cifar100",
            "ImageNet16-120": "imagenet16-120",
        }.get(self.dataset)
        if key is None:
            return {"available": False}
        if self.proxy_backend == "benchmark" and self._api_kind == "full":
            values = np.asarray(
                [
                    self.api.query_test_acc_by_index(index, self.dataset)
                    for index in range(15625)
                ],
                dtype=np.float64,
            )
            return {
                "available": True,
                "metric_name": "NAS-Bench-201 test accuracy",
                "average": float(np.mean(values)),
                "maximum": float(np.max(values)),
                "source": "full NAS-Bench-201 API",
            }
        if not hasattr(self, "nasbench201"):
            path = self.asset_root / "meta_acc_predictor" / "data" / "nasbench201.pt"
            if not path.is_file():
                return {"available": False}
            data = torch.load(str(path), map_location="cpu")
        else:
            data = self.nasbench201
        values = np.asarray(data["test-acc"][key], dtype=np.float64)
        return {
            "available": True,
            "metric_name": "NAS-Bench-201 test accuracy",
            "average": float(np.mean(values)),
            "maximum": float(np.max(values)),
            "source": "compact MetaD2A test labels",
        }
