from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from thesis_pipeline.strong_metrics import metric_suite
from thesis_pipeline.tag_mapping import parse_json_list
from thesis_pipeline.text_normalization import truncate_text


SEARCH_MODULE_PATH = Path(__file__).with_name("08_feature_ablation_and_search.py")


def load_search_module():
    spec = importlib.util.spec_from_file_location("strong_search_module", SEARCH_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SEARCH_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_progress(message: str) -> None:
    path = Path("outputs/report_notes/metric_improvement_progress_log.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().isoformat(timespec='seconds')}: {message}\n")


def parse_params(text: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in str(text).split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def load_features() -> dict[str, np.ndarray]:
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


def valid_search_rows(search: pd.DataFrame) -> pd.DataFrame:
    result = search.copy()
    result["balanced_score"] = pd.to_numeric(result["balanced_score"], errors="coerce")
    result["cluster_count"] = pd.to_numeric(result["cluster_count"], errors="coerce")
    result["largest_cluster_share"] = pd.to_numeric(result["largest_cluster_share"], errors="coerce")
    result["noise_share"] = pd.to_numeric(result.get("noise_share", 0.0), errors="coerce").fillna(0.0)
    result = result[result["balanced_score"].notna()]
    if "error" in result.columns:
        result = result[result["error"].fillna("").astype(str).eq("")]
    valid = result[
        result["cluster_count"].between(8, 25)
        & result["largest_cluster_share"].le(0.40)
        & result["noise_share"].le(0.25)
    ].copy()
    if valid.empty:
        valid = result.copy()
    return valid.sort_values("balanced_score", ascending=False).reset_index(drop=True)


def labels_for_config(row: pd.Series, features: dict[str, np.ndarray], search_module) -> np.ndarray:
    values = search_module.feature_by_name(str(row["feature_set"]), features)
    params = parse_params(row.get("params", ""))
    method = str(row["method"])
    if method == "leiden":
        k = int(float(params.get("k", row.get("k", 30))))
        resolution = float(params.get("resolution", row.get("resolution", 1.0)))
        seed = int(float(params.get("seed", row.get("seed", 42))))
        graph = search_module.build_leiden_graph(values, k, "cosine")
        return search_module.leiden_labels_from_graph(graph, resolution, seed)
    if method == "kmeans":
        n_clusters = int(float(params.get("n_clusters", row.get("n_clusters", 12))))
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(values)
    if method == "agglomerative":
        n_clusters = int(float(params.get("n_clusters", row.get("n_clusters", 12))))
        return AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(values)
    if method == "hdbscan":
        min_cluster_size = int(float(params.get("min_cluster_size", 50)))
        min_samples = int(float(params.get("min_samples", 10)))
        selection = params.get("method", "eom")
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=selection,
        ).fit_predict(values)
    raise ValueError(f"Unsupported method: {method}")


def with_preview(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["text_preview"] = result["text"].map(lambda x: truncate_text(str(x), 240))
    return result


def save_final_umap_html(df: pd.DataFrame) -> None:
    hover = ["id", "year", "month", "tags_raw", "macro_tags", "cluster_final", "text_preview"]
    data = with_preview(df)
    fig2 = px.scatter(
        data,
        x="umap2_x",
        y="umap2_y",
        color=data["cluster_final"].astype(str),
        hover_data=hover,
        title="UMAP 2D colored by final cluster",
    )
    fig2.update_traces(marker={"size": 6, "opacity": 0.75})
    fig2.write_html("outputs/figures/umap2d_final.html", include_plotlyjs="cdn", full_html=True)
    fig3 = px.scatter_3d(
        data,
        x="umap3_x",
        y="umap3_y",
        z="umap3_z",
        color=data["cluster_final"].astype(str),
        hover_data=hover,
        title="UMAP 3D colored by final cluster",
    )
    fig3.update_traces(marker={"size": 4, "opacity": 0.75})
    fig3.write_html("outputs/figures/umap3d_final.html", include_plotlyjs="cdn", full_html=True)


def scatter_png(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str,
    path: str,
    title: str,
    z: str | None = None,
) -> None:
    codes = pd.Categorical(df[label_col].astype(str)).codes
    fig = plt.figure(figsize=(10, 7))
    if z:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df[x], df[y], df[z], c=codes, cmap="tab20", s=7, alpha=0.7, linewidths=0)
        ax.set_zlabel(z)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(df[x], df[y], c=codes, cmap="tab20", s=7, alpha=0.7, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def explode_tag_counts(df: pd.DataFrame, label_col: str, tags_col: str) -> pd.DataFrame:
    rows = []
    for cluster, part in df.groupby(label_col):
        size = len(part)
        counts: dict[str, int] = {}
        for values in part[tags_col].map(parse_json_list):
            for value in values:
                counts[value] = counts.get(value, 0) + 1
        for tag, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            rows.append(
                {
                    "cluster_final": cluster,
                    "tag": tag,
                    "count": int(count),
                    "share_of_cluster": float(count / size) if size else 0.0,
                    "cluster_size": int(size),
                }
            )
    return pd.DataFrame(rows)


def ctfidf_terms(df: pd.DataFrame, label_col: str, top_n: int = 20) -> pd.DataFrame:
    grouped = df.groupby(label_col)["text_clean"].apply(lambda values: " ".join(values.fillna("").astype(str))).reset_index()
    vectorizer = CountVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.90,
        max_features=25000,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    counts = vectorizer.fit_transform(grouped["text_clean"])
    terms = np.asarray(vectorizer.get_feature_names_out())
    term_totals = np.asarray(counts.sum(axis=1)).ravel()
    tf = counts.multiply(1 / np.maximum(term_totals, 1)[:, None])
    df_terms = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((1 + counts.shape[0]) / (1 + df_terms)) + 1
    scores = tf.multiply(idf)
    rows = []
    for i, cluster in enumerate(grouped[label_col]):
        row = scores.getrow(i)
        if row.nnz == 0:
            continue
        order = np.argsort(row.data)[::-1][:top_n]
        for rank, pos in enumerate(order, start=1):
            term_index = row.indices[pos]
            rows.append(
                {
                    "cluster_final": cluster,
                    "rank": rank,
                    "term": terms[term_index],
                    "score": float(row.data[pos]),
                }
            )
    return pd.DataFrame(rows)


def example_tables(df: pd.DataFrame, labels: np.ndarray, features: np.ndarray, per_cluster: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = sorted(pd.unique(labels))
    centroids = []
    for cluster in clusters:
        centroids.append(features[labels == cluster].mean(axis=0))
    distances = pairwise_distances(features, np.vstack(centroids), metric="cosine")
    central_rows = []
    borderline_rows = []
    for col, cluster in enumerate(clusters):
        idx = np.where(labels == cluster)[0]
        own = distances[idx, col]
        central_idx = idx[np.argsort(own)[:per_cluster]]
        other = np.delete(distances[idx], col, axis=1)
        nearest_other = other.min(axis=1) if other.shape[1] else np.ones_like(own)
        margin = nearest_other - own
        border_idx = idx[np.argsort(margin)[:per_cluster]]
        for rank, row_idx in enumerate(central_idx, start=1):
            row = df.iloc[row_idx]
            central_rows.append(example_row(row, cluster, rank, own[np.where(idx == row_idx)[0][0]]))
        for rank, row_idx in enumerate(border_idx, start=1):
            local = np.where(idx == row_idx)[0][0]
            row = df.iloc[row_idx]
            borderline_rows.append(example_row(row, cluster, rank, margin[local]))
    return pd.DataFrame(central_rows), pd.DataFrame(borderline_rows)


def example_row(row: pd.Series, cluster: object, rank: int, score: float) -> dict[str, object]:
    return {
        "cluster_final": cluster,
        "rank": rank,
        "score": float(score),
        "id": row["id"],
        "year": row["year"],
        "month": row["month"],
        "tags_raw": row["tags_raw"],
        "macro_tags": row["macro_tags"],
        "text_preview": truncate_text(str(row["text"]), 300),
    }


def top_join(table: pd.DataFrame, cluster: object, column: str, n: int = 5) -> str:
    part = table[table["cluster_final"].astype(str) == str(cluster)].head(n)
    if part.empty:
        return ""
    return " | ".join(part[column].astype(str).tolist())


def interpretation_cards(
    df: pd.DataFrame,
    macro_counts: pd.DataFrame,
    raw_counts: pd.DataFrame,
    terms: pd.DataFrame,
    central: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    total = len(df)
    for cluster, part in df.groupby("cluster_final"):
        top_macros = top_join(macro_counts, cluster, "tag", 4)
        top_terms = top_join(terms, cluster, "term", 8)
        top_raw = top_join(raw_counts, cluster, "tag", 5)
        central_text = ""
        central_id = ""
        central_part = central[central["cluster_final"].astype(str) == str(cluster)].head(1)
        if not central_part.empty:
            central_id = str(central_part.iloc[0]["id"])
            central_text = str(central_part.iloc[0]["text_preview"])
        rows.append(
            {
                "cluster_final": cluster,
                "size": int(len(part)),
                "share": float(len(part) / total),
                "suggested_name": top_macros or top_terms,
                "top_macro_tags": top_macros,
                "top_raw_tags": top_raw,
                "top_ctfidf_terms": top_terms,
                "year_min": int(part["year"].min()),
                "year_max": int(part["year"].max()),
                "central_example_id": central_id,
                "central_example_preview": central_text,
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def cluster_entropy(df: pd.DataFrame, label_col: str = "cluster_final") -> pd.DataFrame:
    rows = []
    for cluster, part in df.groupby(label_col):
        macro_counts: dict[str, int] = {}
        raw_counts: dict[str, int] = {}
        for values in part["macro_tags"].map(parse_json_list):
            for value in values:
                macro_counts[value] = macro_counts.get(value, 0) + 1
        for values in part["tags_raw"].map(parse_json_list):
            for value in values:
                raw_counts[value] = raw_counts.get(value, 0) + 1
        rows.append(
            {
                "cluster_final": cluster,
                "macro_tag_entropy": entropy_from_counts(macro_counts.values()),
                "raw_tag_entropy": entropy_from_counts(raw_counts.values()),
                "macro_tag_unique": len(macro_counts),
                "raw_tag_unique": len(raw_counts),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_final")


def entropy_from_counts(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=float)
    if not len(values) or values.sum() == 0:
        return 0.0
    shares = values / values.sum()
    return float(-(shares * np.log2(shares)).sum())


def yearly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby(["cluster_final", "year"], as_index=False).size()
    totals = df.groupby("cluster_final").size().rename("cluster_size").reset_index()
    result = result.merge(totals, on="cluster_final", how="left")
    result["share_of_cluster"] = result["size"] / result["cluster_size"]
    return result.sort_values(["cluster_final", "year"])


def structural_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("cluster_final")
        .agg(
            rows=("id", "size"),
            text_length_chars_mean=("text_length_chars", "mean"),
            text_length_chars_median=("text_length_chars", "median"),
            text_length_words_mean=("text_length_words", "mean"),
            text_length_words_median=("text_length_words", "median"),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .reset_index()
    )
    result["share"] = result["rows"] / len(df)
    return result.sort_values("rows", ascending=False)


def report_ready_summary(
    cards: pd.DataFrame,
    entropy: pd.DataFrame,
    structural: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    result = cards.merge(entropy, on="cluster_final", how="left").merge(
        structural[
            [
                "cluster_final",
                "text_length_chars_mean",
                "text_length_words_mean",
                "year_min",
                "year_max",
            ]
        ],
        on="cluster_final",
        how="left",
        suffixes=("", "_structural"),
    )
    final_excl = metrics[(metrics["model"].eq("final")) & (metrics["subset"].eq("excluding_other"))].iloc[0]
    result["final_ari_excluding_other"] = float(final_excl["ari"])
    result["final_v_measure_excluding_other"] = float(final_excl["v_measure"])
    result["final_pairwise_f1"] = float(final_excl["pairwise_f1"])
    return result.sort_values("size", ascending=False)


def save_heatmap(macro_counts: pd.DataFrame) -> None:
    top_tags = macro_counts.groupby("tag")["count"].sum().sort_values(ascending=False).head(18).index
    part = macro_counts[macro_counts["tag"].isin(top_tags)].copy()
    pivot = part.pivot_table(index="cluster_final", columns="tag", values="share_of_cluster", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(13, max(6, len(pivot) * 0.35)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Final clusters by macro-tag share")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig("outputs/figures/cluster_macro_tag_heatmap.png", dpi=180)
    plt.close(fig)


def save_metrics_plot(before_after: pd.DataFrame) -> None:
    rows = before_after[before_after["metric"].isin(["ari_excluding_other", "v_measure_excluding_other", "pairwise_f1_all"])]
    labels = rows["metric"].tolist()
    old_values = rows["old_leiden"].astype(float).tolist()
    final_values = rows["final"].astype(float).tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, old_values, width=0.36, label="old Leiden")
    ax.bar(x + 0.18, final_values, width=0.36, label="final")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, max(final_values + old_values + [0.45]) * 1.15)
    ax.set_title("External metrics before/after final selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/figures/metrics_before_after.png", dpi=180)
    plt.close(fig)


def save_search_plot(search: pd.DataFrame) -> None:
    top = search.sort_values("balanced_score", ascending=False).head(25).copy()
    top["label"] = top["method"].astype(str) + " / " + top["feature_set"].astype(str).str[:32] + " / " + top["params"].astype(str).str[:28]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(np.arange(len(top)), top["balanced_score"].astype(float), color="#4477aa")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("balanced score")
    ax.set_title("Top clustering search configurations")
    fig.tight_layout()
    fig.savefig("outputs/figures/clustering_search_top_configs.png", dpi=180)
    plt.close(fig)


def save_year_coverage(df: pd.DataFrame) -> None:
    counts = df.groupby("year", as_index=False).size()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(counts["year"], counts["size"], color="#228833")
    ax.set_title("Tagged dataset year coverage")
    ax.set_xlabel("year")
    ax.set_ylabel("rows")
    fig.tight_layout()
    fig.savefig("outputs/figures/year_coverage.png", dpi=180)
    plt.close(fig)


def write_final_reports(
    final_row: pd.Series,
    before_after: pd.DataFrame,
    metrics: pd.DataFrame,
    cards: pd.DataFrame,
    search: pd.DataFrame,
) -> None:
    best_raw = search.sort_values("balanced_score", ascending=False).iloc[0]
    selection_lines = [
        "# Final clustering selection",
        "",
        "Selection used tag-derived labels only for validation and ranking; feature vectors and clustering inputs are text-only.",
        "The final model is the highest-scoring valid configuration with 8-25 clusters, largest cluster share <= 0.40, and noise share <= 0.25.",
        "",
        "## Selected configuration",
        pd.DataFrame([final_row.to_dict()]).to_markdown(index=False),
        "",
        "## Before/after metrics",
        before_after.to_markdown(index=False),
        "",
        "## Full metric suite",
        metrics.to_markdown(index=False),
        "",
        "## Best raw score vs final choice",
        "",
        "The best raw balanced-score configuration may differ from the final choice because the final choice enforces the requested interpretable 8-25 cluster range.",
        pd.DataFrame([best_raw.to_dict(), final_row.to_dict()]).to_markdown(index=False),
        "",
        "## Search note",
        f"Search rows evaluated: {len(search)}.",
    ]
    Path("outputs/report_notes/10_final_clustering_selection.md").write_text("\n".join(selection_lines), encoding="utf-8")

    interpretation_lines = [
        "# Final cluster interpretation",
        "",
        "Cluster labels below are interpretive summaries from c-TF-IDF terms, dominant macro-tags, raw tags, and central examples.",
        "They are not used as clustering features.",
        "",
        "## Cluster cards",
        cards.to_markdown(index=False),
    ]
    Path("outputs/report_notes/11_cluster_interpretation.md").write_text("\n".join(interpretation_lines), encoding="utf-8")


def write_executed_notebook(df: pd.DataFrame, final_selection: pd.DataFrame, metrics: pd.DataFrame) -> None:
    import nbformat
    from nbclient import NotebookClient

    Path("notebooks").mkdir(parents=True, exist_ok=True)
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Tagged corpus final clustering run\n\nExecuted artifact for the thesis pipeline."),
        nbformat.v4.new_code_cell(
            "\n".join(
                [
                    "import json, platform, subprocess, sys",
                    "import pandas as pd",
                    "print('Python:', sys.version.split()[0])",
                    "print('Platform:', platform.platform())",
                    "try:",
                    "    import torch",
                    "    print('Torch:', torch.__version__)",
                    "    print('CUDA available:', torch.cuda.is_available())",
                    "    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')",
                    "except Exception as exc:",
                    "    print('Torch/GPU check failed:', type(exc).__name__, exc)",
                    "try:",
                    "    print('Git commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())",
                    "except Exception as exc:",
                    "    print('Git commit unavailable:', type(exc).__name__, exc)",
                ]
            )
        ),
        nbformat.v4.new_code_cell(
            "\n".join(
                [
                    "df = pd.read_csv('data/processed/anekdots_tagged_clustered.csv')",
                    "print({'rows': len(df), 'final_clusters': int(df['cluster_final'].nunique()), 'largest_cluster_share': float(df['cluster_final'].value_counts(normalize=True).max())})",
                    "df[['id', 'year', 'month', 'cluster_old_leiden', 'cluster_final', 'macro_tags']].head()",
                ]
            )
        ),
        nbformat.v4.new_code_cell(
            "\n".join(
                [
                    "selection = pd.read_csv('outputs/tables/final_clustering_selection.csv')",
                    "selection",
                ]
            )
        ),
        nbformat.v4.new_code_cell(
            "\n".join(
                [
                    "metrics = pd.read_csv('outputs/tables/final_metrics_summary.csv')",
                    "metrics[['model', 'subset', 'rows', 'cluster_count', 'ari', 'ami', 'v_measure', 'pairwise_f1', 'largest_cluster_share']]",
                ]
            )
        ),
        nbformat.v4.new_code_cell(
            "\n".join(
                [
                    "from pathlib import Path",
                    "for path in [",
                    "    'outputs/figures/umap2d_final.html',",
                    "    'outputs/figures/umap3d_final.html',",
                    "    'outputs/figures/umap2d_final.png',",
                    "    'outputs/figures/umap3d_final.png',",
                    "    'outputs/tables/cluster_final_interpretation_cards.csv',",
                    "]:",
                    "    p = Path(path)",
                    "    print(path, p.exists(), p.stat().st_size if p.exists() else 0)",
                ]
            )
        ),
    ]
    selection_dict = final_selection.iloc[0].to_dict() if not final_selection.empty else {}
    nb.metadata["codex_execution_note"] = json.loads(json.dumps(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rows": int(len(df)),
            "final_selection": selection_dict,
        },
        default=str,
    ))
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "."}})
    client.execute()
    nbformat.write(nb, "notebooks/tagged_corpus_analysis_executed_colab.ipynb")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/anekdots_tagged_clustered.csv")
    parser.add_argument("--search", default="outputs/tables/clustering_search_all_runs.csv")
    parser.add_argument("--skip-notebook", action="store_true")
    args = parser.parse_args()

    for folder in ["outputs/tables", "outputs/figures", "outputs/report_notes", "notebooks"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    search = pd.read_csv(args.search)
    search_valid = valid_search_rows(search)
    search_module = load_search_module()
    features = load_features()

    final_row = search_valid.iloc[0].copy()
    final_labels = labels_for_config(final_row, features, search_module)
    final_feature_set = str(final_row["feature_set"])
    final_values = search_module.feature_by_name(final_feature_set, features)

    if "cluster_old_leiden" not in df.columns:
        df["cluster_old_leiden"] = df["cluster_leiden"]
    df["cluster_final"] = final_labels
    df["cluster_method"] = str(final_row["method"])
    df["feature_set"] = final_feature_set

    method_columns = {
        "leiden": "cluster_leiden_tuned",
        "kmeans": "cluster_kmeans_best",
        "hdbscan": "cluster_hdbscan_best",
        "agglomerative": "cluster_agglomerative_best",
    }
    chosen_by_method: dict[str, pd.Series] = {}
    for method, column in method_columns.items():
        part = search_valid[search_valid["method"].astype(str).eq(method)]
        if part.empty:
            continue
        row = part.iloc[0].copy()
        df[column] = labels_for_config(row, features, search_module)
        chosen_by_method[column] = row

    df.to_csv(args.dataset, index=False, encoding="utf-8")

    metric_frames = []
    label_configs = [("old_leiden", df["cluster_old_leiden"].to_numpy(), {"method": "leiden", "feature_set": "dense_bge_pca", "params": "k=30;resolution=1.0;seed=42"})]
    for column, row in chosen_by_method.items():
        label_configs.append((column, df[column].to_numpy(), row.to_dict()))
    label_configs.append(("final", df["cluster_final"].to_numpy(), final_row.to_dict()))
    for model, labels, config in label_configs:
        suite = metric_suite(df, labels)
        suite.insert(0, "model", model)
        suite.insert(1, "method", config.get("method", ""))
        suite.insert(2, "feature_set", config.get("feature_set", ""))
        suite.insert(3, "params", config.get("params", ""))
        metric_frames.append(suite)
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv("outputs/tables/final_metrics_summary.csv", index=False, encoding="utf-8")

    final_excl = metrics[(metrics["model"].eq("final")) & (metrics["subset"].eq("excluding_other"))].iloc[0]
    old_excl = metrics[(metrics["model"].eq("old_leiden")) & (metrics["subset"].eq("excluding_other"))].iloc[0]
    before_after = pd.DataFrame(
        [
            {
                "metric": metric,
                "old_leiden": float(old_excl[metric]),
                "final": float(final_excl[metric]),
                "delta": float(final_excl[metric] - old_excl[metric]),
            }
            for metric in ["ari", "ami", "v_measure", "pairwise_f1", "pairwise_precision", "pairwise_recall", "largest_cluster_share"]
        ]
    ).rename(columns={"ari": "ari_excluding_other"})
    before_after.loc[before_after["metric"].eq("ari"), "metric"] = "ari_excluding_other"
    before_after.loc[before_after["metric"].eq("v_measure"), "metric"] = "v_measure_excluding_other"
    before_after.loc[before_after["metric"].eq("pairwise_f1"), "metric"] = "pairwise_f1_all"
    before_after.to_csv("outputs/tables/metrics_before_after.csv", index=False, encoding="utf-8")

    final_selection = pd.DataFrame([final_row.to_dict()])
    final_selection["final_rows"] = len(df)
    final_selection["final_cluster_count"] = int(pd.Series(final_labels).nunique())
    final_selection["final_largest_cluster_share"] = float(pd.Series(final_labels).value_counts(normalize=True).max())
    final_selection["delta_ari_vs_old_leiden"] = float(final_excl["ari"] - old_excl["ari"])
    final_selection["delta_v_measure_vs_old_leiden"] = float(final_excl["v_measure"] - old_excl["v_measure"])
    final_selection["delta_pairwise_f1_vs_old_leiden"] = float(final_excl["pairwise_f1"] - old_excl["pairwise_f1"])
    final_selection.to_csv("outputs/tables/final_clustering_selection.csv", index=False, encoding="utf-8")

    cluster_sizes = df["cluster_final"].value_counts().rename_axis("cluster_final").reset_index(name="size")
    cluster_sizes["share"] = cluster_sizes["size"] / len(df)
    cluster_sizes.to_csv("outputs/tables/cluster_final_sizes.csv", index=False, encoding="utf-8")
    macro_counts = explode_tag_counts(df, "cluster_final", "macro_tags")
    raw_counts = explode_tag_counts(df, "cluster_final", "tags_raw")
    macro_counts.to_csv("outputs/tables/cluster_final_macro_tag_matrix.csv", index=False, encoding="utf-8")
    raw_counts.to_csv("outputs/tables/cluster_final_raw_tag_matrix.csv", index=False, encoding="utf-8")
    terms = ctfidf_terms(df, "cluster_final")
    terms.to_csv("outputs/tables/cluster_final_ctfidf_terms.csv", index=False, encoding="utf-8")
    terms.to_csv("outputs/tables/cluster_ctfidf_terms.csv", index=False, encoding="utf-8")
    central, borderline = example_tables(df, final_labels, final_values)
    central.to_csv("outputs/tables/cluster_final_central_examples.csv", index=False, encoding="utf-8")
    borderline.to_csv("outputs/tables/cluster_final_borderline_examples.csv", index=False, encoding="utf-8")
    cards = interpretation_cards(df, macro_counts, raw_counts, terms, central)
    cards.to_csv("outputs/tables/cluster_final_interpretation_cards.csv", index=False, encoding="utf-8")
    cards.to_csv("outputs/tables/cluster_interpretation_cards.csv", index=False, encoding="utf-8")
    entropy = cluster_entropy(df)
    years = yearly_distribution(df)
    structural = structural_summary(df)
    summary = report_ready_summary(cards, entropy, structural, metrics)
    entropy.to_csv("outputs/tables/cluster_entropy.csv", index=False, encoding="utf-8")
    years.to_csv("outputs/tables/cluster_yearly_distribution.csv", index=False, encoding="utf-8")
    structural.to_csv("outputs/tables/cluster_structural_summary.csv", index=False, encoding="utf-8")
    summary.to_csv("outputs/tables/cluster_report_ready_summary.csv", index=False, encoding="utf-8")

    save_final_umap_html(df)
    scatter_png(df, "umap2_x", "umap2_y", "cluster_final", "outputs/figures/umap2d_final.png", "UMAP 2D colored by final cluster")
    scatter_png(df, "umap3_x", "umap3_y", "cluster_final", "outputs/figures/umap3d_final.png", "UMAP 3D colored by final cluster", z="umap3_z")
    pca = np.load("data/embeddings/tagged_pca128.npy")
    pca_df = df.copy()
    pca_df["pca_x"], pca_df["pca_y"], pca_df["pca_z"] = pca[:, 0], pca[:, 1], pca[:, 2]
    scatter_png(pca_df, "pca_x", "pca_y", "cluster_old_leiden", "outputs/figures/pca3d_baseline.png", "PCA 3D baseline colored by old Leiden", z="pca_z")
    save_heatmap(macro_counts)
    save_metrics_plot(before_after)
    save_search_plot(search)
    save_year_coverage(df)
    write_final_reports(final_row, before_after, metrics, cards, search)
    if not args.skip_notebook:
        write_executed_notebook(df, final_selection, metrics)

    append_progress(
        "Phase 6-8 final selection complete: "
        f"method={final_row['method']}, feature={final_feature_set}, "
        f"clusters={pd.Series(final_labels).nunique()}, ARI={final_excl['ari']:.3f}, "
        f"V={final_excl['v_measure']:.3f}, F1={final_excl['pairwise_f1']:.3f}."
    )
    print(
        json.dumps(
            {
                "method": str(final_row["method"]),
                "feature_set": final_feature_set,
                "clusters": int(pd.Series(final_labels).nunique()),
                "ari": float(final_excl["ari"]),
                "v_measure": float(final_excl["v_measure"]),
                "pairwise_f1": float(final_excl["pairwise_f1"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
