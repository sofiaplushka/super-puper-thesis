from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from thesis_pipeline.tag_mapping import parse_json_list


@dataclass(frozen=True)
class PairwiseCounts:
    total_pairs: int
    model_positive_pairs: int
    label_positive_pairs: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float


def macro_sets(series: Iterable[object]) -> list[set[str]]:
    return [set(parse_json_list(value)) for value in series]


def primary_labels(series: Iterable[object]) -> list[str]:
    labels = []
    for values in macro_sets(series):
        labels.append(sorted(values)[0] if values else "other")
    return labels


def mask_excluding_other(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(["other" not in tags and bool(tags) for tags in macro_sets(df["macro_tags"])])


def mask_single_clear_label(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([len(tags) == 1 and "other" not in tags for tags in macro_sets(df["macro_tags"])])


def external_metrics(labels_true: Iterable[object], labels_pred: Iterable[object]) -> dict[str, float]:
    y_true = list(labels_true)
    y_pred = list(labels_pred)
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return {
            "ari": 0.0,
            "ami": 0.0,
            "nmi": 0.0,
            "homogeneity": 0.0,
            "completeness": 0.0,
            "v_measure": 0.0,
        }
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
    }


def masks_to_ints(tag_sets: list[set[str]]) -> tuple[list[int], dict[str, int]]:
    tags = sorted(set().union(*tag_sets) if tag_sets else set())
    lookup = {tag: i for i, tag in enumerate(tags)}
    values = []
    for tag_set in tag_sets:
        mask = 0
        for tag in tag_set:
            mask |= 1 << lookup[tag]
        values.append(mask)
    return values, lookup


def shared_label_pair_count(mask_counts: Counter[int]) -> int:
    items = [(mask, count) for mask, count in mask_counts.items() if mask]
    total = 0
    for i, (mask_a, count_a) in enumerate(items):
        if count_a >= 2:
            total += count_a * (count_a - 1) // 2
        for mask_b, count_b in items[i + 1 :]:
            if mask_a & mask_b:
                total += count_a * count_b
    return int(total)


def exact_pairwise_multilabel(labels_pred: Iterable[object], tag_sets: list[set[str]]) -> PairwiseCounts:
    pred = list(labels_pred)
    masks, _ = masks_to_ints(tag_sets)
    n = len(pred)
    total_pairs = n * (n - 1) // 2
    label_positive = shared_label_pair_count(Counter(masks))
    model_positive = 0
    true_positive = 0
    by_cluster: dict[object, list[int]] = defaultdict(list)
    for label, mask in zip(pred, masks):
        by_cluster[label].append(mask)
    for cluster_masks in by_cluster.values():
        size = len(cluster_masks)
        model_positive += size * (size - 1) // 2
        true_positive += shared_label_pair_count(Counter(cluster_masks))
    false_positive = model_positive - true_positive
    false_negative = label_positive - true_positive
    true_negative = total_pairs - true_positive - false_positive - false_negative
    precision = true_positive / model_positive if model_positive else 0.0
    recall = true_positive / label_positive if label_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PairwiseCounts(
        total_pairs=int(total_pairs),
        model_positive_pairs=int(model_positive),
        label_positive_pairs=int(label_positive),
        true_positive=int(true_positive),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_negative=int(true_negative),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def cluster_size_entropy(labels_pred: Iterable[object]) -> float:
    counts = np.asarray(list(Counter(labels_pred).values()), dtype=float)
    shares = counts / counts.sum()
    return float(-np.sum([p * math.log(p, 2) for p in shares if p > 0]))


def metric_row(
    df: pd.DataFrame,
    labels_pred: Iterable[object],
    subset_name: str,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    labels = np.asarray(list(labels_pred))
    part = df if mask is None else df.loc[mask].copy()
    part_labels = labels if mask is None else labels[mask]
    true_labels = primary_labels(part["macro_tags"])
    metrics = external_metrics(true_labels, part_labels)
    pairwise = exact_pairwise_multilabel(part_labels, macro_sets(part["macro_tags"]))
    counts = pd.Series(part_labels).value_counts(normalize=True)
    row: dict[str, object] = {
        "subset": subset_name,
        "rows": int(len(part)),
        "cluster_count": int(len(set(part_labels))),
        "largest_cluster_share": float(counts.max()) if len(counts) else 0.0,
        "cluster_size_entropy": cluster_size_entropy(part_labels),
        **metrics,
        "pairwise_precision": pairwise.precision,
        "pairwise_recall": pairwise.recall,
        "pairwise_f1": pairwise.f1,
        "pairwise_total_pairs": pairwise.total_pairs,
        "pairwise_model_positive_pairs": pairwise.model_positive_pairs,
        "pairwise_label_positive_pairs": pairwise.label_positive_pairs,
        "pairwise_true_positive": pairwise.true_positive,
        "pairwise_false_positive": pairwise.false_positive,
        "pairwise_false_negative": pairwise.false_negative,
        "pairwise_true_negative": pairwise.true_negative,
    }
    return row


def metric_suite(df: pd.DataFrame, labels_pred: Iterable[object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            metric_row(df, labels_pred, "all"),
            metric_row(df, labels_pred, "excluding_other", mask_excluding_other(df)),
            metric_row(df, labels_pred, "single_clear_label", mask_single_clear_label(df)),
        ]
    )

