from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from thesis_pipeline.strong_metrics import macro_sets, metric_row


def append_progress(message: str) -> None:
    path = Path("outputs/report_notes/metric_improvement_progress_log.md")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().isoformat(timespec='seconds')}: {message}\n")


def structural_matrix(df: pd.DataFrame) -> np.ndarray:
    text = df["text"].fillna("").astype(str)
    mat = pd.DataFrame(
        {
            "chars": text.map(len),
            "words": text.map(lambda x: len(x.split())),
            "lines": text.map(lambda x: max(1, x.count("\n") + 1)),
            "dash_count": text.map(lambda x: x.count("-") + x.count("—")),
            "question_count": text.map(lambda x: x.count("?")),
            "exclamation_count": text.map(lambda x: x.count("!")),
            "comma_count": text.map(lambda x: x.count(",")),
            "quote_count": text.map(lambda x: x.count('"') + x.count("«") + x.count("»")),
            "dialogue_marker": text.map(lambda x: int((x.count("-") + x.count("—")) >= 2)),
            "uppercase_ratio": text.map(lambda x: sum(1 for c in x if c.isupper()) / max(1, sum(1 for c in x if c.isalpha()))),
        }
    )
    return normalize(StandardScaler().fit_transform(mat), norm="l2")


def svd_tfidf(texts: list[str], analyzer: str, ngram_range: tuple[int, int], max_features: int, seed: int) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    n_components = min(128, max(2, matrix.shape[1] - 1))
    values = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(matrix)
    return normalize(values, norm="l2").astype("float32", copy=False)


def save_features(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    Path("data/features").mkdir(parents=True, exist_ok=True)
    texts = df["text_clean" if "text_clean" in df.columns else "text"].fillna("").astype(str).tolist()
    dense = normalize(np.load("data/embeddings/tagged_pca128.npy"), norm="l2").astype("float32", copy=False)
    word = svd_tfidf(texts, "word", (1, 2), 12000, seed)
    char = svd_tfidf(texts, "char_wb", (3, 5), 16000, seed)
    structural = structural_matrix(df).astype("float32", copy=False)
    lexical = normalize(np.hstack([word, char]), norm="l2").astype("float32", copy=False)
    hybrid = normalize(np.hstack([dense, lexical * 0.75, structural * 0.25]), norm="l2").astype("float32", copy=False)
    features = {
        "dense_bge_pca": dense,
        "tfidf_word_svd": word,
        "tfidf_char_svd": char,
        "structural_features": structural,
        "hybrid_dense_lexical": normalize(np.hstack([dense, lexical]), norm="l2").astype("float32", copy=False),
        "hybrid_dense_lexical_structural": hybrid,
    }
    for name, values in features.items():
        np.save(f"data/features/{name}.npy", values)
    return features | {"_lexical": lexical}


def balanced_score(row: dict[str, object]) -> float:
    score = (
        0.30 * float(row.get("ari_excluding_other", 0))
        + 0.25 * float(row.get("v_measure_excluding_other", 0))
        + 0.20 * float(row.get("ami_excluding_other", 0))
        + 0.20 * float(row.get("pairwise_f1_all", 0))
    )
    cluster_count = int(row.get("cluster_count", 0))
    largest = float(row.get("largest_cluster_share", 1))
    noise = float(row.get("noise_share", 0))
    if cluster_count < 6 or cluster_count > 30:
        score -= 0.15
    if largest > 0.40:
        score -= (largest - 0.40)
    if noise > 0.25:
        score -= (noise - 0.25)
    return float(score)


def evaluate_labels(df: pd.DataFrame, labels: Iterable[object], run: dict[str, object]) -> dict[str, object]:
    labels_arr = np.asarray(list(labels))
    all_row = metric_row(df, labels_arr, "all")
    excl_row = metric_row(df, labels_arr, "excluding_other")
    single_row = metric_row(df, labels_arr, "single_clear_label")
    counts = pd.Series(labels_arr).value_counts(normalize=True)
    row = dict(run)
    row.update(
        {
            "rows": len(df),
            "cluster_count": int(len(set(labels_arr))),
            "largest_cluster_share": float(counts.max()),
            "noise_share": float((labels_arr == -1).mean()) if np.issubdtype(labels_arr.dtype, np.number) else 0.0,
            "ari_all": all_row["ari"],
            "ami_all": all_row["ami"],
            "v_measure_all": all_row["v_measure"],
            "ari_excluding_other": excl_row["ari"],
            "ami_excluding_other": excl_row["ami"],
            "v_measure_excluding_other": excl_row["v_measure"],
            "ari_single_clear": single_row["ari"],
            "ami_single_clear": single_row["ami"],
            "v_measure_single_clear": single_row["v_measure"],
            "pairwise_f1_all": all_row["pairwise_f1"],
            "pairwise_precision_all": all_row["pairwise_precision"],
            "pairwise_recall_all": all_row["pairwise_recall"],
        }
    )
    row["balanced_score"] = balanced_score(row)
    return row


def quick_feature_ablation(df: pd.DataFrame, features: dict[str, np.ndarray], seed: int) -> pd.DataFrame:
    rows = []
    base_names = [k for k in features if not k.startswith("_")]
    for name in base_names:
        labels = MiniBatchKMeans(n_clusters=12, random_state=seed, n_init=10, batch_size=512).fit_predict(features[name])
        rows.append(evaluate_labels(df, labels, {"feature_set": name, "method": "minibatch_kmeans_ablation", "params": "n_clusters=12"}))
    dense = features["dense_bge_pca"]
    lexical = features["_lexical"]
    structural = features["structural_features"]
    for dw in [0.5, 0.75, 1.0, 1.25]:
        for lw in [0.25, 0.5, 0.75, 1.0, 1.25]:
            values = normalize(np.hstack([dense * dw, lexical * lw]), norm="l2")
            labels = MiniBatchKMeans(n_clusters=12, random_state=seed, n_init=5, batch_size=512).fit_predict(values)
            name = f"hybrid_dense_lexical_dw{dw}_lw{lw}"
            rows.append(evaluate_labels(df, labels, {"feature_set": name, "method": "minibatch_kmeans_ablation", "params": f"dense={dw};lexical={lw}"}))
            for sw in [0.0, 0.1, 0.25, 0.5]:
                if sw == 0.0:
                    continue
                values_s = normalize(np.hstack([dense * dw, lexical * lw, structural * sw]), norm="l2")
                labels_s = MiniBatchKMeans(n_clusters=12, random_state=seed, n_init=5, batch_size=512).fit_predict(values_s)
                name_s = f"hybrid_dense_lexical_structural_dw{dw}_lw{lw}_sw{sw}"
                rows.append(evaluate_labels(df, labels_s, {"feature_set": name_s, "method": "minibatch_kmeans_ablation", "params": f"dense={dw};lexical={lw};structural={sw}"}))
    result = pd.DataFrame(rows).sort_values("balanced_score", ascending=False)
    result.to_csv("outputs/tables/feature_ablation_metrics.csv", index=False, encoding="utf-8")
    return result


def feature_by_name(name: str, features: dict[str, np.ndarray]) -> np.ndarray:
    if name in features:
        return features[name]
    dense = features["dense_bge_pca"]
    lexical = features["_lexical"]
    structural = features["structural_features"]
    import re

    dw = float(re.search(r"dw([0-9.]+)", name).group(1))
    lw = float(re.search(r"lw([0-9.]+)", name).group(1))
    sw_match = re.search(r"sw([0-9.]+)", name)
    if sw_match:
        sw = float(sw_match.group(1))
        return normalize(np.hstack([dense * dw, lexical * lw, structural * sw]), norm="l2").astype("float32", copy=False)
    return normalize(np.hstack([dense * dw, lexical * lw]), norm="l2").astype("float32", copy=False)


def build_leiden_graph(values: np.ndarray, k: int, metric: str):
    import igraph as ig

    k = min(k, max(2, len(values) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(values)
    distances, indices = nn.kneighbors(values)
    edges = []
    weights = []
    for i in range(len(values)):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            a, b = sorted((int(i), int(j)))
            edges.append((a, b))
            if metric == "cosine":
                weights.append(float(max(0.0, 1.0 - dist)))
            else:
                weights.append(float(1.0 / (1.0 + dist)))
    graph = ig.Graph(n=len(values), edges=edges, directed=False)
    graph.es["weight"] = weights
    graph.simplify(combine_edges={"weight": "mean"})
    return graph


def leiden_labels_from_graph(graph, resolution: float, seed: int) -> np.ndarray:
    import leidenalg

    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(part.membership)


def leiden_labels(values: np.ndarray, k: int, resolution: float, seed: int, metric: str) -> np.ndarray:
    return leiden_labels_from_graph(build_leiden_graph(values, k, metric), resolution, seed)


def strong_search(df: pd.DataFrame, features: dict[str, np.ndarray], ablation: pd.DataFrame, seed: int) -> pd.DataFrame:
    output_path = Path("outputs/tables/clustering_search_all_runs.csv")
    partial_path = Path("outputs/tables/clustering_search_partial.csv")
    selected = list(dict.fromkeys(list(ablation["feature_set"].head(3)) + ["dense_bge_pca", "tfidf_word_svd", "tfidf_char_svd"]))
    hdbscan_selected = set(selected[:2] + ["dense_bge_pca"])
    rows = []

    def flush() -> None:
        if rows:
            pd.DataFrame(rows).sort_values("balanced_score", ascending=False).to_csv(partial_path, index=False, encoding="utf-8")

    def add_row(row: dict[str, object]) -> None:
        rows.append(row)
        if len(rows) % 25 == 0:
            flush()

    for name in selected:
        values = feature_by_name(name, features)
        for k in [10, 15, 20, 30, 50, 75]:
            try:
                graph = build_leiden_graph(values, k, "cosine")
            except Exception as exc:
                graph = None
                add_row({"feature_set": name, "method": "leiden", "params": f"k={k};graph_build", "error": f"{type(exc).__name__}: {exc}", "balanced_score": -999})
            for resolution in [0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
                for run_seed in [1, 7, 42]:
                    try:
                        if graph is None:
                            continue
                        labels = leiden_labels_from_graph(graph, resolution, run_seed)
                        add_row(
                            evaluate_labels(
                                df,
                                labels,
                                {
                                    "feature_set": name,
                                    "method": "leiden",
                                    "metric": "cosine",
                                    "params": f"k={k};resolution={resolution};seed={run_seed}",
                                    "k": k,
                                    "resolution": resolution,
                                    "seed": run_seed,
                                },
                            )
                        )
                    except Exception as exc:
                        add_row({"feature_set": name, "method": "leiden", "params": f"k={k};resolution={resolution};seed={run_seed}", "error": f"{type(exc).__name__}: {exc}", "balanced_score": -999})
        for n_clusters in [6, 8, 10, 12, 15, 18, 20, 25, 30]:
            labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(values)
            add_row(evaluate_labels(df, labels, {"feature_set": name, "method": "kmeans", "params": f"n_clusters={n_clusters}", "n_clusters": n_clusters}))
        for n_clusters in [8, 10, 12, 15, 20, 25]:
            try:
                labels = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(values)
                add_row(evaluate_labels(df, labels, {"feature_set": name, "method": "agglomerative", "metric": "cosine", "params": f"n_clusters={n_clusters};linkage=average"}))
            except Exception as exc:
                add_row({"feature_set": name, "method": "agglomerative", "params": f"n_clusters={n_clusters}", "error": f"{type(exc).__name__}: {exc}", "balanced_score": -999})
        if name in hdbscan_selected:
            for min_cluster_size, min_samples, method in [
                (30, 5, "eom"),
                (50, 10, "eom"),
                (75, 10, "eom"),
                (100, 20, "eom"),
                (150, 20, "eom"),
                (50, 10, "leaf"),
            ]:
                try:
                    labels = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric="euclidean",
                        cluster_selection_method=method,
                    ).fit_predict(values)
                    add_row(
                        evaluate_labels(
                            df,
                            labels,
                            {
                                "feature_set": name,
                                "method": "hdbscan",
                                "metric": "euclidean",
                                "params": f"min_cluster_size={min_cluster_size};min_samples={min_samples};method={method}",
                            },
                        )
                    )
                except Exception as exc:
                    add_row({"feature_set": name, "method": "hdbscan", "params": f"min_cluster_size={min_cluster_size};min_samples={min_samples};method={method}", "error": f"{type(exc).__name__}: {exc}", "balanced_score": -999})
        flush()
    result = pd.DataFrame(rows).sort_values("balanced_score", ascending=False)
    result.to_csv(output_path, index=False, encoding="utf-8")
    result[result["method"] == "leiden"].to_csv("outputs/tables/leiden_grid_search_ranked.csv", index=False, encoding="utf-8")
    result[result["method"] != "leiden"].to_csv("outputs/tables/baseline_grid_search_ranked.csv", index=False, encoding="utf-8")
    result.groupby("method", as_index=False).head(5).to_csv("outputs/tables/clustering_baselines_comparison.csv", index=False, encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    return result


def write_reports(ablation: pd.DataFrame, search: pd.DataFrame) -> None:
    Path("outputs/report_notes/08_feature_ablation.md").write_text(
        "\n".join(
            [
                "# Feature ablation",
                "",
                "All feature sets are derived only from joke text or non-label structural text properties.",
                "No tag, macro-tag, primary label, or cluster label is included in any feature vector.",
                "",
                "## Top ablation configurations",
                ablation.head(20).to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )
    Path("outputs/report_notes/09_strong_clustering_search.md").write_text(
        "\n".join(
            [
                "# Strong clustering search",
                "",
                "The search ranks configurations by a balanced score using excluding-other ARI/AMI/V-measure,",
                "exact pairwise F1, cluster-count constraints, largest-cluster penalty, and HDBSCAN noise penalty.",
                "Tags are used only for validation and ranking, not for feature construction.",
                "",
                "## Top configurations",
                search.head(25).to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered", default="data/processed/anekdots_tagged_clustered.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/report_notes").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.clustered)
    features = save_features(df, args.seed)
    ablation = quick_feature_ablation(df, features, args.seed)
    append_progress(
        f"Phase 4 feature ablation complete: best={ablation.iloc[0]['feature_set']} "
        f"score={ablation.iloc[0]['balanced_score']:.3f}."
    )
    search = strong_search(df, features, ablation, args.seed)
    append_progress(
        f"Phase 5 clustering search complete: best={search.iloc[0]['method']} "
        f"{search.iloc[0]['feature_set']} score={search.iloc[0]['balanced_score']:.3f}, "
        f"ARI_ex_other={search.iloc[0]['ari_excluding_other']:.3f}, "
        f"V={search.iloc[0]['v_measure_excluding_other']:.3f}, F1={search.iloc[0]['pairwise_f1_all']:.3f}."
    )
    write_reports(ablation, search)
    print(search.head(10)[["feature_set", "method", "params", "cluster_count", "largest_cluster_share", "ari_excluding_other", "v_measure_excluding_other", "pairwise_f1_all", "balanced_score"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
