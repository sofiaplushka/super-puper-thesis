from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)

from thesis_pipeline.clustering import leiden_cluster
from thesis_pipeline.tag_mapping import parse_json_list


def explode_counts(df: pd.DataFrame, column: str, label_name: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for value in parse_json_list(row[column]):
            rows.append({"id": row["id"], label_name: value})
    return pd.DataFrame(rows)


def tag_distribution(df: pd.DataFrame, column: str, label_name: str) -> pd.DataFrame:
    exploded = explode_counts(df, column, label_name)
    if exploded.empty:
        return pd.DataFrame(columns=[label_name, "count", "share"])
    counts = (
        exploded[label_name]
        .value_counts()
        .rename_axis(label_name)
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / len(df)
    return counts


def tag_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["tag_count", "macro_tag_count"]].copy()
    return (
        out.value_counts()
        .rename("rows")
        .reset_index()
        .sort_values(["tag_count", "macro_tag_count"])
    )


def cluster_tag_matrix(df: pd.DataFrame, tag_column: str) -> pd.DataFrame:
    clusters = sorted(df["cluster_leiden"].unique())
    tags = sorted(
        {tag for values in df[tag_column].map(parse_json_list) for tag in values}
    )
    matrix = pd.DataFrame(0, index=clusters, columns=tags)
    for _, row in df.iterrows():
        for tag in parse_json_list(row[tag_column]):
            matrix.loc[row["cluster_leiden"], tag] += 1
    matrix.index.name = "cluster_leiden"
    return matrix.reset_index()


def purity_entropy(matrix: pd.DataFrame, prefix: str) -> pd.DataFrame:
    tag_cols = [c for c in matrix.columns if c != "cluster_leiden"]
    rows = []
    for _, row in matrix.iterrows():
        counts = row[tag_cols].to_numpy(dtype=float)
        total = counts.sum()
        if total == 0:
            rows.append(
                {
                    "cluster_leiden": row["cluster_leiden"],
                    f"dominant_{prefix}_tag": None,
                    f"dominant_{prefix}_share": 0,
                    f"{prefix}_entropy": None,
                }
            )
            continue
        shares = counts / total
        max_idx = int(np.argmax(shares))
        entropy = -float(np.sum([p * math.log(p, 2) for p in shares if p > 0]))
        rows.append(
            {
                "cluster_leiden": row["cluster_leiden"],
                f"dominant_{prefix}_tag": tag_cols[max_idx],
                f"dominant_{prefix}_share": float(shares[max_idx]),
                f"{prefix}_entropy": entropy,
            }
        )
    return pd.DataFrame(rows)


def external_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "adjusted_rand_index": float(adjusted_rand_score(y_true, y_pred)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
    }


def pairwise_multilabel_metrics(
    df: pd.DataFrame, seed: int, max_pairs: int = 1_000_000
) -> pd.DataFrame:
    n = len(df)
    total_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    labels = df["cluster_leiden"].to_numpy()
    tag_sets = [set(parse_json_list(v)) for v in df["macro_tags"]]
    if total_pairs <= max_pairs:
        pairs = itertools.combinations(range(n), 2)
        sampled = total_pairs
        exact = True
    else:
        sampled_pairs = set()
        while len(sampled_pairs) < max_pairs:
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n - 1))
            if b >= a:
                b += 1
            sampled_pairs.add(tuple(sorted((a, b))))
        pairs = iter(sampled_pairs)
        sampled = len(sampled_pairs)
        exact = False
    tp = fp = fn = tn = 0
    for i, j in pairs:
        model_pos = labels[i] == labels[j]
        label_pos = bool(tag_sets[i].intersection(tag_sets[j]))
        if model_pos and label_pos:
            tp += 1
        elif model_pos and not label_pos:
            fp += 1
        elif not model_pos and label_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return pd.DataFrame(
        [
            {
                "exact": exact,
                "total_possible_pairs": total_pairs,
                "pairs_evaluated": sampled,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ]
    )


def internal_metrics(
    features: np.ndarray, labels: np.ndarray, seed: int, modularity: float | None = None
) -> pd.DataFrame:
    unique = np.unique(labels)
    if len(unique) < 2:
        return pd.DataFrame(
            [
                {
                    "metric": "not_computable",
                    "value": None,
                    "reason": "fewer than two clusters",
                }
            ]
        )
    sample_size = min(len(features), 3000)
    silhouette = silhouette_score(
        features,
        labels,
        metric="cosine",
        sample_size=sample_size if sample_size < len(features) else None,
        random_state=seed,
    )
    rows = [
        {
            "metric": "silhouette_cosine",
            "value": float(silhouette),
            "space": "embedding_or_pca",
        },
        {
            "metric": "calinski_harabasz",
            "value": float(calinski_harabasz_score(features, labels)),
            "space": "pca",
        },
        {
            "metric": "davies_bouldin",
            "value": float(davies_bouldin_score(features, labels)),
            "space": "pca",
        },
        {"metric": "modularity", "value": modularity, "space": "knn_graph"},
        {"metric": "cluster_count", "value": int(len(unique)), "space": "labels"},
        {
            "metric": "largest_cluster_share",
            "value": float(pd.Series(labels).value_counts(normalize=True).max()),
            "space": "labels",
        },
    ]
    return pd.DataFrame(rows)


def stability_grid(
    features: np.ndarray, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [42, 123, 2026] if len(features) <= 5000 else [42, 123]
    grid_rows = []
    partitions: dict[str, np.ndarray] = {}
    for k in [15, 30, 50]:
        for resolution in [0.5, 1.0, 1.5]:
            for run_seed in seeds:
                result = leiden_cluster(
                    features, k=k, resolution=resolution, seed=run_seed
                )
                key = f"k{k}_r{resolution}_s{run_seed}"
                partitions[key] = result.labels
                counts = pd.Series(result.labels).value_counts(normalize=True)
                grid_rows.append(
                    {
                        "run_id": key,
                        "k": k,
                        "resolution": resolution,
                        "seed": run_seed,
                        "cluster_count": int(len(np.unique(result.labels))),
                        "largest_cluster_share": float(counts.max()),
                        "modularity": result.modularity,
                        "fallback": result.used_fallback,
                    }
                )
    pair_rows = []
    keys = list(partitions)
    for a, b in itertools.combinations(keys, 2):
        pair_rows.append(
            {
                "run_a": a,
                "run_b": b,
                "ari": float(adjusted_rand_score(partitions[a], partitions[b])),
                "ami": float(adjusted_mutual_info_score(partitions[a], partitions[b])),
            }
        )
    return pd.DataFrame(grid_rows), pd.DataFrame(pair_rows)


def save_heatmap(matrix: pd.DataFrame, path: str | Path) -> None:
    indexed = matrix.set_index("cluster_leiden")
    shares = indexed.div(indexed.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig = px.imshow(
        shares,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Cluster x macro tag row-normalized shares",
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
