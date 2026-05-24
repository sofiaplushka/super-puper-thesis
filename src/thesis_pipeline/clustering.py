from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from thesis_pipeline.tag_mapping import parse_json_list
from thesis_pipeline.text_normalization import truncate_text


@dataclass(frozen=True)
class LeidenResult:
    labels: np.ndarray
    modularity: float | None
    used_fallback: bool


def load_feature_matrix(embeddings_path: str | Path, pca_path: str | Path | None = None) -> np.ndarray:
    if pca_path and Path(pca_path).exists():
        return np.load(pca_path).astype("float32", copy=False)
    return np.load(embeddings_path).astype("float32", copy=False)


def build_knn_graph(features: np.ndarray, k: int):
    import igraph as ig

    n = len(features)
    k = min(k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    edges = []
    weights = []
    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            if i == j:
                continue
            a, b = sorted((int(i), int(j)))
            edges.append((a, b))
            weights.append(float(max(0.0, 1.0 - dist)))
    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.es["weight"] = weights
    graph.simplify(combine_edges={"weight": "mean"})
    return graph


def leiden_cluster(features: np.ndarray, k: int, resolution: float, seed: int) -> LeidenResult:
    try:
        import leidenalg
    except Exception:
        labels = KMeans(n_clusters=max(2, min(12, len(features) // 40)), random_state=seed, n_init=10).fit_predict(features)
        return LeidenResult(labels=np.asarray(labels), modularity=None, used_fallback=True)
    graph = build_knn_graph(features, k)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    modularity = graph.modularity(partition.membership, weights=graph.es["weight"])
    return LeidenResult(labels=np.asarray(partition.membership, dtype=int), modularity=float(modularity), used_fallback=False)


def run_umap(features: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, max(2, len(features) - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(features).astype("float32", copy=False)


def cluster_size_table(labels: np.ndarray) -> pd.DataFrame:
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return pd.DataFrame(
        {
            "cluster_leiden": values,
            "size": counts,
            "share": counts / total,
        }
    ).sort_values("size", ascending=False)


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, part in df.groupby("cluster_leiden"):
        raw_counts: dict[str, int] = {}
        macro_counts: dict[str, int] = {}
        for raw_values in part["tags_raw"].map(parse_json_list):
            for value in raw_values:
                raw_counts[value] = raw_counts.get(value, 0) + 1
        for macro_values in part["macro_tags"].map(parse_json_list):
            for value in macro_values:
                macro_counts[value] = macro_counts.get(value, 0) + 1
        rows.append(
            {
                "cluster_leiden": cluster,
                "size": len(part),
                "dominant_raw_tag": max(raw_counts, key=raw_counts.get) if raw_counts else None,
                "dominant_raw_tag_count": max(raw_counts.values()) if raw_counts else 0,
                "dominant_macro_tag": max(macro_counts, key=macro_counts.get) if macro_counts else None,
                "dominant_macro_tag_count": max(macro_counts.values()) if macro_counts else 0,
                "year_min": int(part["year"].min()),
                "year_max": int(part["year"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def central_and_borderline_examples(df: pd.DataFrame, features: np.ndarray, per_cluster: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = df["cluster_leiden"].to_numpy()
    centroids = {}
    for cluster in sorted(np.unique(labels)):
        centroids[cluster] = features[labels == cluster].mean(axis=0)
    centroid_matrix = np.vstack([centroids[c] for c in sorted(centroids)])
    cluster_order = list(sorted(centroids))
    distances = pairwise_distances(features, centroid_matrix, metric="cosine")
    central_rows = []
    borderline_rows = []
    for cluster in cluster_order:
        idx = np.where(labels == cluster)[0]
        col = cluster_order.index(cluster)
        own = distances[idx, col]
        central_idx = idx[np.argsort(own)[:per_cluster]]
        other = np.delete(distances[idx], col, axis=1)
        nearest_other = other.min(axis=1) if other.shape[1] else np.ones_like(own)
        margin = nearest_other - own
        borderline_idx = idx[np.argsort(margin)[:per_cluster]]
        for rank, row_idx in enumerate(central_idx, start=1):
            central_rows.append(example_record(df.iloc[row_idx], rank, own[np.where(idx == row_idx)[0][0]]))
        for rank, row_idx in enumerate(borderline_idx, start=1):
            local = np.where(idx == row_idx)[0][0]
            borderline_rows.append(example_record(df.iloc[row_idx], rank, margin[local]))
    return pd.DataFrame(central_rows), pd.DataFrame(borderline_rows)


def example_record(row: pd.Series, rank: int, score: float) -> dict[str, object]:
    return {
        "cluster_leiden": row["cluster_leiden"],
        "rank": rank,
        "score": float(score),
        "id": row["id"],
        "year": row["year"],
        "month": row["month"],
        "tags_raw": row["tags_raw"],
        "macro_tags": row["macro_tags"],
        "text_preview": truncate_text(row["text"], 300),
    }


def pca3d(features: np.ndarray, seed: int) -> np.ndarray:
    return PCA(n_components=3, random_state=seed).fit_transform(features).astype("float32", copy=False)

