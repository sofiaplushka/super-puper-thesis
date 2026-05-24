from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from thesis_pipeline.clustering import (
    central_and_borderline_examples,
    cluster_size_table,
    leiden_cluster,
    load_feature_matrix,
    pca3d,
    run_umap,
    summarize_clusters,
)
from thesis_pipeline.visualization import save_pca3d, save_umap2d, save_umap3d


def write_report(path: Path, df: pd.DataFrame, args, result, sizes: pd.DataFrame) -> None:
    imbalance = float(sizes["share"].max() / sizes["share"].median()) if len(sizes) else 0.0
    lines = [
        "# 3D UMAP and Leiden clustering",
        "",
        f"Dataset: `{args.dataset}`",
        f"Embeddings: `{args.embeddings}`",
        f"PCA input: `{args.pca}`",
        f"kNN k: `{args.k}`",
        f"Leiden resolution: `{args.resolution}`",
        f"Seed: `{args.seed}`",
        f"Leiden fallback used: `{result.used_fallback}`",
        f"Modularity: `{result.modularity}`",
        f"Number of clusters: **{df['cluster_leiden'].nunique()}**",
        f"Largest/median cluster share ratio: `{imbalance:.3f}`",
        "",
        "Clustering is performed in BGE-M3/PCA embedding space. UMAP-2D and UMAP-3D are visualization",
        "layers only and are not used as clustering inputs.",
        "",
        "## Cluster sizes",
        sizes.to_markdown(index=False),
        "",
        "## Copy-ready practical section notes",
        "The updated practical pipeline uses only jokes that have at least one anekdot.ru tag.",
        "Embeddings are computed before any dimensional visualization, and Leiden clustering is applied",
        "to the high-dimensional embedding/PCA representation rather than to UMAP coordinates.",
        "The 3D UMAP figure is therefore a visual projection of the same Leiden labels used in the 2D plot.",
        "This distinction is important because UMAP can change local visual geometry while preserving only",
        "an approximate neighborhood structure.",
        "Cluster summaries, central examples, and borderline examples were exported to CSV so the written",
        "thesis can discuss both strong and weak cluster interpretations without rerunning the notebook.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/anekdots_tagged.csv")
    parser.add_argument("--embeddings", default="data/embeddings/tagged_bge_m3.npy")
    parser.add_argument("--pca", default="data/embeddings/tagged_pca128.npy")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    args = parser.parse_args()

    for folder in ["data/processed", "data/embeddings", "outputs/figures", "outputs/tables", "outputs/report_notes"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    features = load_feature_matrix(args.embeddings, args.pca)
    if len(df) != len(features):
        raise SystemExit("Dataset and feature matrix row counts do not match.")
    result = leiden_cluster(features, args.k, args.resolution, args.seed)
    df["cluster_leiden"] = result.labels
    try:
        from sklearn.cluster import KMeans

        n_clusters = df["cluster_leiden"].nunique()
        df["cluster_kmeans"] = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(features)
    except Exception:
        pass
    umap2 = run_umap(features, 2, args.umap_neighbors, args.umap_min_dist, args.seed)
    umap3 = run_umap(features, 3, args.umap_neighbors, args.umap_min_dist, args.seed)
    np.save("data/embeddings/tagged_umap2d.npy", umap2)
    np.save("data/embeddings/tagged_umap3d.npy", umap3)
    df["umap2_x"], df["umap2_y"] = umap2[:, 0], umap2[:, 1]
    df["umap3_x"], df["umap3_y"], df["umap3_z"] = umap3[:, 0], umap3[:, 1], umap3[:, 2]
    df.to_csv("data/processed/anekdots_tagged_clustered.csv", index=False, encoding="utf-8")

    sizes = cluster_size_table(result.labels)
    summary = summarize_clusters(df)
    central, borderline = central_and_borderline_examples(df, features, per_cluster=10)
    sizes.to_csv("outputs/tables/cluster_sizes.csv", index=False, encoding="utf-8")
    summary.to_csv("outputs/tables/cluster_summary.csv", index=False, encoding="utf-8")
    central.to_csv("outputs/tables/cluster_central_examples.csv", index=False, encoding="utf-8")
    borderline.to_csv("outputs/tables/cluster_borderline_examples.csv", index=False, encoding="utf-8")
    save_umap2d(df, "outputs/figures/umap2d_leiden.html")
    save_umap3d(df, "outputs/figures/umap3d_leiden.html")
    save_pca3d(df, pca3d(features, args.seed), "outputs/figures/pca3d_baseline.html")
    write_report(Path("outputs/report_notes/02_3d_umap_and_clustering.md"), df, args, result, sizes)
    print({"rows": len(df), "clusters": int(df["cluster_leiden"].nunique()), "modularity": result.modularity})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

