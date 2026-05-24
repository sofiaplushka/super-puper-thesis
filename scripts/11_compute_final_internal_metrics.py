from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize


SEARCH_MODULE_PATH = Path(__file__).with_name("08_feature_ablation_and_search.py")


def load_search_module():
    spec = importlib.util.spec_from_file_location("strong_search_module", SEARCH_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SEARCH_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_params(text: object) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in str(text).split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip()] = value.strip()
    return params


def load_existing_features() -> dict[str, np.ndarray]:
    features = {
        "dense_bge_pca": normalize(np.load("data/features/dense_bge_pca.npy"), norm="l2").astype("float32", copy=False),
        "tfidf_word_svd": normalize(np.load("data/features/tfidf_word_svd.npy"), norm="l2").astype("float32", copy=False),
        "tfidf_char_svd": normalize(np.load("data/features/tfidf_char_svd.npy"), norm="l2").astype("float32", copy=False),
        "structural_features": normalize(np.load("data/features/structural_features.npy"), norm="l2").astype("float32", copy=False),
        "hybrid_dense_lexical": normalize(np.load("data/features/hybrid_dense_lexical.npy"), norm="l2").astype("float32", copy=False),
        "hybrid_dense_lexical_structural": normalize(np.load("data/features/hybrid_dense_lexical_structural.npy"), norm="l2").astype("float32", copy=False),
    }
    features["_lexical"] = normalize(
        np.hstack([features["tfidf_word_svd"], features["tfidf_char_svd"]]),
        norm="l2",
    ).astype("float32", copy=False)
    return features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clustered", default="data/processed/anekdots_tagged_clustered.csv")
    parser.add_argument("--selection", default="outputs/tables/final_clustering_selection.csv")
    parser.add_argument("--output", default="outputs/tables/final_internal_cluster_metrics.csv")
    parser.add_argument("--legacy-output", default="outputs/tables/internal_cluster_metrics.csv")
    parser.add_argument("--archive-initial", default="outputs/tables/internal_cluster_metrics_initial_leiden.csv")
    args = parser.parse_args()

    output = Path(args.output)
    legacy_output = Path(args.legacy_output)
    archive_initial = Path(args.archive_initial)
    output.parent.mkdir(parents=True, exist_ok=True)

    if legacy_output.exists() and not archive_initial.exists():
        shutil.copyfile(legacy_output, archive_initial)

    df = pd.read_csv(args.clustered)
    selection = pd.read_csv(args.selection).iloc[0]
    params = parse_params(selection["params"])
    method = str(selection["method"])
    feature_set = str(selection["feature_set"])
    k = int(float(params.get("k", 75)))
    resolution = float(params.get("resolution", 2.0))
    seed = int(float(params.get("seed", 7)))
    labels = df["cluster_final"].to_numpy()
    search_module = load_search_module()
    features = search_module.feature_by_name(feature_set, load_existing_features())

    cluster_count = int(pd.Series(labels).nunique())
    largest_cluster_share = float(pd.Series(labels).value_counts(normalize=True).max())
    silhouette = float(silhouette_score(features, labels, metric="cosine")) if cluster_count > 1 else np.nan
    calinski = float(calinski_harabasz_score(features, labels)) if cluster_count > 1 else np.nan
    davies = float(davies_bouldin_score(features, labels)) if cluster_count > 1 else np.nan
    modularity: float | str
    modularity_note = ""
    try:
        graph = search_module.build_leiden_graph(features, k, "cosine")
        modularity = float(graph.modularity(labels.tolist(), weights=graph.es["weight"]))
    except Exception as exc:
        modularity = "not_recomputed"
        modularity_note = f"{type(exc).__name__}: {exc}"

    row = {
        "method": method,
        "feature_set": feature_set,
        "k": k,
        "resolution": resolution,
        "seed": seed,
        "cluster_label_column": "cluster_final",
        "cluster_count": cluster_count,
        "largest_cluster_share": largest_cluster_share,
        "silhouette_cosine": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        "modularity": modularity,
        "modularity_note": modularity_note,
        "graph_k": k,
        "graph_metric": "cosine",
        "feature_space": "hybrid BGE/PCA + lexical TF-IDF/SVD",
    }
    result = pd.DataFrame([row])
    result.to_csv(output, index=False, encoding="utf-8")
    result.to_csv(legacy_output, index=False, encoding="utf-8")
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
