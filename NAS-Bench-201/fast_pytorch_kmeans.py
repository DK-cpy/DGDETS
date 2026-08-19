"""Dependency-compatible deterministic torch KMeans fallback."""

from __future__ import absolute_import, division, print_function

import torch


class KMeans(object):
    def __init__(self, n_clusters, mode="euclidean", max_iter=20, **kwargs):
        if mode != "euclidean":
            raise ValueError("only euclidean mode is supported")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.centroids = None

    def fit_predict(self, values):
        x = values.reshape(values.shape[0], -1).float()
        if x.shape[0] < self.n_clusters:
            raise ValueError("n_clusters exceeds the number of samples")
        # Deterministic farthest-point initialization.
        selected = [0]
        minimum = torch.full(
            (x.shape[0],), float("inf"), device=x.device, dtype=x.dtype
        )
        for _ in range(1, self.n_clusters):
            distance = torch.sum((x - x[selected[-1]]) ** 2, dim=1)
            minimum = torch.minimum(minimum, distance)
            selected.append(int(torch.argmax(minimum).item()))
        centers = x[torch.tensor(selected, device=x.device)].clone()
        labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for _ in range(self.max_iter):
            distance = torch.cdist(x, centers)
            new_labels = torch.argmin(distance, dim=1)
            if torch.equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = []
            for cluster in range(self.n_clusters):
                members = x[labels == cluster]
                new_centers.append(
                    members.mean(dim=0) if len(members) else centers[cluster]
                )
            centers = torch.stack(new_centers)
        self.centroids = centers.reshape(
            (self.n_clusters,) + tuple(values.shape[1:])
        )
        return labels

