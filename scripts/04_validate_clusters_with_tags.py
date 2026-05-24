from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from thesis_pipeline.clustering import load_feature_matrix
from thesis_pipeline.tag_mapping import (
    load_macro_mapping,
    macro_lookup,
    normalize_tag,
    parse_json_list,
)
from thesis_pipeline.validation import (
    cluster_tag_matrix,
    external_metrics,
    internal_metrics,
    pairwise_multilabel_metrics,
    purity_entropy,
    save_heatmap,
    stability_grid,
    tag_count_distribution,
    tag_distribution,
)


def unmapped_tags(df: pd.DataFrame, tag_map: str) -> pd.DataFrame:
    lookup = macro_lookup(load_macro_mapping(tag_map))
    counts = {}
    for tags in df["tags_raw"].map(parse_json_list):
        for tag in tags:
            if normalize_tag(tag) not in lookup:
                counts[tag] = counts.get(tag, 0) + 1
    result = pd.DataFrame([{"tag_raw": k, "count": v} for k, v in counts.items()])
    if result.empty:
        return pd.DataFrame(columns=["tag_raw", "count"])
    return result.sort_values("count", ascending=False)


def single_label_metrics(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["macro_tags"].map(lambda x: len(parse_json_list(x)) == 1)
    part = df.loc[mask]
    if part.empty:
        return pd.DataFrame()
    metrics = external_metrics(
        part["macro_tags"].map(lambda x: parse_json_list(x)[0]), part["cluster_leiden"]
    )
    metrics["rows"] = int(len(part))
    return pd.DataFrame([metrics])


def primary_macro_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = external_metrics(df["primary_macro_tag"], df["cluster_leiden"])
    metrics["rows"] = int(len(df))
    metrics["label_source"] = "primary_macro_tag_all_rows_weaker_approximation"
    return pd.DataFrame([metrics])


def write_report(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Cluster validation with site tags",
        "",
        "Tags are external silver labels from anekdot.ru, not manual gold labels. Multi-label jokes",
        "are handled explicitly through cluster-tag matrices and pairwise multilabel metrics.",
        "",
        "## Single-label external metrics",
        (
            tables["single"].to_markdown(index=False)
            if not tables["single"].empty
            else "No single-label rows."
        ),
        "",
        "## Primary macro approximation metrics",
        tables["primary"].to_markdown(index=False),
        "",
        "## Pairwise multilabel metrics",
        tables["pairwise"].to_markdown(index=False),
        "",
        "## Internal metrics",
        tables["internal"].to_markdown(index=False),
        "",
        "## Interpretation",
        "High homogeneity would mean that clusters rarely mix macro tags, while high completeness would",
        "mean that a macro tag is concentrated inside one cluster. For humor data both values are expected",
        "to be imperfect because tags are noisy, broad, and multi-label. Pairwise F1 is a stricter view:",
        "it checks whether pairs placed in the same cluster also share at least one macro tag.",
        "",
        "## Copy-ready report paragraphs",
        "The updated validation treats anekdot.ru tags as silver labels. This avoids manual row labeling",
        "but also means the evaluation measures agreement with the site's topic taxonomy rather than",
        "absolute semantic truth.",
        "Because jokes may have several tags, the report includes both single-label metrics on a filtered",
        "subset and pairwise multilabel precision/recall/F1 on the full dataset.",
        "Internal metrics are computed in embedding/PCA space and never on UMAP coordinates.",
        "Stability is checked by varying kNN graph size, Leiden resolution, and random seeds.",
        "",
        "## Suggested captions",
        "Table: External cluster quality metrics against single macro tags.",
        "Table: Pairwise multilabel cluster agreement using shared macro tags.",
        "Figure: Row-normalized cluster by macro-tag heatmap.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clustered", default="data/processed/anekdots_tagged_clustered.csv"
    )
    parser.add_argument("--embeddings", default="data/embeddings/tagged_bge_m3.npy")
    parser.add_argument("--pca", default="data/embeddings/tagged_pca128.npy")
    parser.add_argument("--tag-map", default="config/tag_macro_categories.yml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for folder in ["outputs/tables", "outputs/figures", "outputs/report_notes"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clustered)
    features = load_feature_matrix(args.embeddings, args.pca)
    labels = df["cluster_leiden"].to_numpy()

    tag_distribution(df, "tags_raw", "tag_raw").to_csv(
        "outputs/tables/tag_distribution_raw.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    tag_distribution(df, "macro_tags", "macro_tag").to_csv(
        "outputs/tables/tag_distribution_macro.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    tag_count_distribution(df).to_csv(
        "outputs/tables/tag_count_distribution.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    unmapped_tags(df, args.tag_map).to_csv(
        "outputs/tables/unmapped_tags.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    raw_matrix = cluster_tag_matrix(df, "tags_raw")
    macro_matrix = cluster_tag_matrix(df, "macro_tags")
    raw_matrix.to_csv(
        "outputs/tables/cluster_raw_tag_matrix.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    macro_matrix.to_csv(
        "outputs/tables/cluster_macro_tag_matrix.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    purity = purity_entropy(raw_matrix, "raw").merge(
        purity_entropy(macro_matrix, "macro"), on="cluster_leiden", how="outer"
    )
    purity.to_csv(
        "outputs/tables/cluster_tag_purity_entropy.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    save_heatmap(macro_matrix, "outputs/figures/cluster_macro_tag_heatmap.html")

    single = single_label_metrics(df)
    primary = primary_macro_metrics(df)
    pairwise = pairwise_multilabel_metrics(df, args.seed)
    internal = internal_metrics(features, labels, args.seed)
    grid, pair_grid = stability_grid(features, args.seed)
    single.to_csv(
        "outputs/tables/external_metrics_single_label.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    primary.to_csv(
        "outputs/tables/external_metrics_primary_macro.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    pairwise.to_csv(
        "outputs/tables/pairwise_multilabel_metrics.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    internal.to_csv(
        "outputs/tables/internal_cluster_metrics.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    grid.to_csv(
        "outputs/tables/leiden_stability_grid.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    pair_grid.to_csv(
        "outputs/tables/leiden_stability_pairwise.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    write_report(
        Path("outputs/report_notes/03_cluster_validation.md"),
        {
            "single": single,
            "primary": primary,
            "pairwise": pairwise,
            "internal": internal,
        },
    )
    print({"rows": len(df), "clusters": int(df["cluster_leiden"].nunique())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
