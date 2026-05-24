from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yaml

from thesis_pipeline.strong_metrics import exact_pairwise_multilabel, external_metrics
from thesis_pipeline.tag_mapping import json_list, parse_json_list


def load_hierarchy(path: str | Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    level1_to_level2: dict[str, list[str]] = {}
    level2_to_level1: dict[str, str] = {}
    for level1, spec in data.items():
        tags = [str(tag) for tag in (spec or {}).get("level2", [])]
        level1_to_level2[str(level1)] = tags
        for tag in tags:
            if tag in level2_to_level1:
                raise ValueError(
                    f"Macro tag {tag!r} appears in more than one level-1 category"
                )
            level2_to_level1[tag] = str(level1)
    return level1_to_level2, level2_to_level1


def primary_from_sets(tag_sets: Iterable[set[str]]) -> list[str]:
    return [sorted(tags)[0] if tags else "other" for tags in tag_sets]


def single_clear_mask(tag_sets: list[set[str]]) -> np.ndarray:
    return np.asarray([len(tags) == 1 and "other" not in tags for tags in tag_sets])


def metric_row(
    labels_pred: np.ndarray,
    tag_sets: list[set[str]],
    label_level: str,
    subset: str,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    if mask is None:
        labels = labels_pred
        sets = tag_sets
    else:
        labels = labels_pred[mask]
        sets = [tag_set for tag_set, keep in zip(tag_sets, mask) if keep]
    metrics = external_metrics(primary_from_sets(sets), labels)
    pairwise = exact_pairwise_multilabel(labels, sets)
    counts = pd.Series(labels).value_counts(normalize=True)
    return {
        "label_level": label_level,
        "subset": subset,
        "rows": int(len(labels)),
        "label_count": int(len(set().union(*sets)) if sets else 0),
        "cluster_count": int(len(set(labels))),
        "largest_cluster_share": float(counts.max()) if len(counts) else 0.0,
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


def level1_sets(level2_sets: list[set[str]], lookup: dict[str, str]) -> list[set[str]]:
    missing = sorted({tag for tags in level2_sets for tag in tags if tag not in lookup})
    if missing:
        raise ValueError(f"Macro tags missing from hierarchy: {missing}")
    return [{lookup[tag] for tag in tags if tag in lookup} for tags in level2_sets]


def write_note(summary: pd.DataFrame, output: str | Path) -> None:
    rows = {
        (row["label_level"], row["subset"]): row
        for row in summary.to_dict(orient="records")
    }
    l1 = rows[("level_1_broad", "all")]
    l2 = rows[("level_2_detailed", "all")]
    l1_single = rows[("level_1_broad", "single_clear_label")]
    l2_single = rows[("level_2_detailed", "single_clear_label")]
    level1_higher = (
        l1["v_measure"] > l2["v_measure"] and l1["pairwise_f1"] > l2["pairwise_f1"]
    )
    observed_note = (
        "In this run the broad level-1 metrics are higher than the level-2 metrics."
        if level1_higher
        else "In this run the broad level-1 metrics are not uniformly higher: broad categories merge many heterogeneous detailed themes, which lowers completeness and increases the number of label-positive pairs in pairwise evaluation."
    )
    text = f"""# Hierarchical macro-tag evaluation

This experiment keeps the final unsupervised Leiden clustering unchanged and
changes only the evaluation target.

The existing macro-tags are treated as level-2 detailed categories. They are
also mapped to broader level-1 categories in `config/macro_tag_hierarchy.yml`.
Level-1 metrics are often expected to be higher because they evaluate broader
thematic agreement: confusing two detailed categories inside the same broad
topic is less severe than confusing unrelated broad topics. This expectation is
not automatic for all metrics. {observed_note}

## Results

| Evaluation target | Subset | ARI | AMI | NMI | Homogeneity | Completeness | V-measure | Pairwise F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Level-1 broad | all | {l1['ari']:.4f} | {l1['ami']:.4f} | {l1['nmi']:.4f} | {l1['homogeneity']:.4f} | {l1['completeness']:.4f} | {l1['v_measure']:.4f} | {l1['pairwise_f1']:.4f} |
| Level-2 detailed | all | {l2['ari']:.4f} | {l2['ami']:.4f} | {l2['nmi']:.4f} | {l2['homogeneity']:.4f} | {l2['completeness']:.4f} | {l2['v_measure']:.4f} | {l2['pairwise_f1']:.4f} |
| Level-1 broad | single-clear-label | {l1_single['ari']:.4f} | {l1_single['ami']:.4f} | {l1_single['nmi']:.4f} | {l1_single['homogeneity']:.4f} | {l1_single['completeness']:.4f} | {l1_single['v_measure']:.4f} | {l1_single['pairwise_f1']:.4f} |
| Level-2 detailed | single-clear-label | {l2_single['ari']:.4f} | {l2_single['ami']:.4f} | {l2_single['nmi']:.4f} | {l2_single['homogeneity']:.4f} | {l2_single['completeness']:.4f} | {l2_single['v_measure']:.4f} | {l2_single['pairwise_f1']:.4f} |

## Interpretation

The level-2 rows should remain the main detailed validation view, because they
compare clusters with the full macro-tag taxonomy. The level-1 rows answer a
different question: whether clusters preserve coarse thematic structure. If
level-1 numbers are higher, they can help explain broad thematic agreement. If
they are lower, as in this local run for V-measure and pairwise F1, the honest
interpretation is that the chosen broad categories are too heterogeneous for the
current cluster structure.
"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clustered", default="data/processed/anekdots_tagged_clustered.csv"
    )
    parser.add_argument("--hierarchy", default="config/macro_tag_hierarchy.yml")
    parser.add_argument(
        "--output", default="outputs/tables/hierarchical_metrics_summary.csv"
    )
    parser.add_argument(
        "--note", default="outputs/report_notes/13_hierarchical_evaluation.md"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.clustered)
    labels = df["cluster_final"].to_numpy()
    _, lookup = load_hierarchy(args.hierarchy)
    level2 = [set(parse_json_list(value)) for value in df["macro_tags"]]
    level1 = level1_sets(level2, lookup)

    rows = []
    for name, tag_sets in [("level_1_broad", level1), ("level_2_detailed", level2)]:
        rows.append(metric_row(labels, tag_sets, name, "all"))
        rows.append(
            metric_row(
                labels,
                tag_sets,
                name,
                "single_clear_label",
                single_clear_mask(tag_sets),
            )
        )
    summary = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")

    enriched = df[["id", "macro_tags"]].copy()
    enriched["level_1_tags"] = [json_list(sorted(tags)) for tags in level1]
    enriched.to_csv(
        "outputs/tables/hierarchical_tag_assignments.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    write_note(summary, args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
