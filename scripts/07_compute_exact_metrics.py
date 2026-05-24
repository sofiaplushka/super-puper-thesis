from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from thesis_pipeline.clustering import build_knn_graph, load_feature_matrix
from thesis_pipeline.strong_metrics import (
    exact_pairwise_multilabel,
    external_metrics,
    macro_sets,
    mask_excluding_other,
    mask_single_clear_label,
    metric_row,
    primary_labels,
)


def append_progress(message: str) -> None:
    path = Path("outputs/report_notes/metric_improvement_progress_log.md")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().isoformat(timespec='seconds')}: {message}\n")


def internal_rows(features: np.ndarray, labels: np.ndarray, seed: int, k: int) -> pd.DataFrame:
    rows = []
    if len(np.unique(labels)) >= 2:
        sample_size = min(3000, len(features))
        rows.extend(
            [
                {
                    "metric": "silhouette_cosine",
                    "value": float(
                        silhouette_score(
                            features,
                            labels,
                            metric="cosine",
                            sample_size=sample_size if sample_size < len(features) else None,
                            random_state=seed,
                        )
                    ),
                    "space": "embedding_or_pca",
                },
                {"metric": "calinski_harabasz", "value": float(calinski_harabasz_score(features, labels)), "space": "pca"},
                {"metric": "davies_bouldin", "value": float(davies_bouldin_score(features, labels)), "space": "pca"},
            ]
        )
    try:
        graph = build_knn_graph(features, k)
        rows.append({"metric": "modularity", "value": float(graph.modularity(labels.tolist(), weights=graph.es["weight"])), "space": "knn_graph"})
    except Exception as exc:
        rows.append({"metric": "modularity", "value": None, "space": "knn_graph", "reason": f"{type(exc).__name__}: {exc}"})
    counts = pd.Series(labels).value_counts(normalize=True)
    rows.extend(
        [
            {"metric": "cluster_count", "value": int(len(np.unique(labels))), "space": "labels"},
            {"metric": "largest_cluster_share", "value": float(counts.max()), "space": "labels"},
        ]
    )
    return pd.DataFrame(rows)


def write_metric_tables(df: pd.DataFrame, labels: np.ndarray, features: np.ndarray, args) -> dict[str, pd.DataFrame]:
    masks = {
        "all": np.ones(len(df), dtype=bool),
        "excluding_other": mask_excluding_other(df),
        "single_clear_label": mask_single_clear_label(df),
        "multi_label": df["macro_tags"].map(lambda x: len(macro_sets([x])[0]) > 1).to_numpy(),
    }
    tables = {}
    for subset, mask in masks.items():
        row = metric_row(df, labels, subset, mask if subset != "all" else None)
        table = pd.DataFrame([row])
        tables[subset] = table
        name = "metrics_multi_label" if subset == "multi_label" else f"metrics_{subset}"
        table.to_csv(f"outputs/tables/{name}.csv", index=False, encoding="utf-8")
    pairwise = exact_pairwise_multilabel(labels, macro_sets(df["macro_tags"]))
    pairwise_df = pd.DataFrame([pairwise.__dict__ | {"exact": True}])
    pairwise_df.to_csv("outputs/tables/pairwise_multilabel_metrics_exact.csv", index=False, encoding="utf-8")
    pairwise_df.to_csv("outputs/tables/pairwise_multilabel_metrics.csv", index=False, encoding="utf-8")
    internal = internal_rows(features, labels, args.seed, args.k)
    internal.to_csv("outputs/tables/internal_cluster_metrics.csv", index=False, encoding="utf-8")
    # Backward-compatible legacy metric files.
    single_mask = masks["single_clear_label"]
    single_ext = external_metrics(primary_labels(df.loc[single_mask, "macro_tags"]), labels[single_mask])
    pd.DataFrame([single_ext | {"rows": int(single_mask.sum())}]).to_csv(
        "outputs/tables/external_metrics_single_label.csv", index=False, encoding="utf-8"
    )
    all_ext = external_metrics(primary_labels(df["macro_tags"]), labels)
    pd.DataFrame([all_ext | {"rows": int(len(df)), "label_source": "primary_macro_tag_all_rows_after_remap"}]).to_csv(
        "outputs/tables/external_metrics_primary_macro.csv", index=False, encoding="utf-8"
    )
    return tables | {"pairwise": pairwise_df, "internal": internal}


def write_before_after(tables: dict[str, pd.DataFrame]) -> None:
    baseline = {
        "single_clear_ari": 0.133207,
        "single_clear_ami": 0.262534,
        "single_clear_v_measure": 0.265831,
        "pairwise_f1_sampled": 0.258864,
        "other_occurrences": 1830.0,
    }
    current = {
        "single_clear_ari": float(tables["single_clear_label"]["ari"].iloc[0]),
        "single_clear_ami": float(tables["single_clear_label"]["ami"].iloc[0]),
        "single_clear_v_measure": float(tables["single_clear_label"]["v_measure"].iloc[0]),
        "excluding_other_ari": float(tables["excluding_other"]["ari"].iloc[0]),
        "excluding_other_v_measure": float(tables["excluding_other"]["v_measure"].iloc[0]),
        "pairwise_f1_exact": float(tables["pairwise"]["f1"].iloc[0]),
        "other_occurrences": 0.0,
    }
    rows = []
    for metric in sorted(set(baseline) | set(current)):
        before = baseline.get(metric)
        after = current.get(metric)
        rows.append(
            {
                "metric": metric,
                "before_baseline": before,
                "after_current": after,
                "delta": (after - before) if before is not None and after is not None else None,
                "note": "baseline pairwise was sampled" if metric == "pairwise_f1_sampled" else "",
            }
        )
    pd.DataFrame(rows).to_csv("outputs/tables/metrics_before_after.csv", index=False, encoding="utf-8")


def write_report(tables: dict[str, pd.DataFrame], labels_col: str) -> None:
    lines = [
        "# Exact validation metrics",
        "",
        f"Cluster label column: `{labels_col}`.",
        "Pairwise multilabel metrics are exact and use combinatorial counting over macro-tag sets.",
        "",
        "## All rows",
        tables["all"].to_markdown(index=False),
        "",
        "## Excluding other",
        tables["excluding_other"].to_markdown(index=False),
        "",
        "## Single clear label",
        tables["single_clear_label"].to_markdown(index=False),
        "",
        "## Multi-label rows",
        tables["multi_label"].to_markdown(index=False),
        "",
        "## Exact pairwise",
        tables["pairwise"].to_markdown(index=False),
        "",
        "The current improvement at this checkpoint is mostly due to semantic macro-tag remapping, not",
        "changed clustering. Feature ablations and clustering search are run in later phases.",
    ]
    Path("outputs/report_notes/07_validation_metrics_exact.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered", default="data/processed/anekdots_tagged_clustered.csv")
    parser.add_argument("--embeddings", default="data/embeddings/tagged_bge_m3.npy")
    parser.add_argument("--pca", default="data/embeddings/tagged_pca128.npy")
    parser.add_argument("--label-column", default="cluster_leiden")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=30)
    args = parser.parse_args()
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.clustered)
    labels = df[args.label_column].to_numpy()
    features = load_feature_matrix(args.embeddings, args.pca)
    tables = write_metric_tables(df, labels, features, args)
    write_before_after(tables)
    write_report(tables, args.label_column)
    append_progress(
        "Phase 3 exact metrics complete: "
        f"excluding_other ARI={tables['excluding_other']['ari'].iloc[0]:.3f}, "
        f"V={tables['excluding_other']['v_measure'].iloc[0]:.3f}, "
        f"exact pairwise F1={tables['pairwise']['f1'].iloc[0]:.3f}."
    )
    print(tables["excluding_other"].to_dict("records")[0])
    print(tables["pairwise"].to_dict("records")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

