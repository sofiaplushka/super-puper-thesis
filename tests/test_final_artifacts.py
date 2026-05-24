from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_OUTPUTS = [
    "outputs/tables/final_clustering_selection.csv",
    "outputs/tables/final_metrics_summary.csv",
    "outputs/tables/final_internal_cluster_metrics.csv",
    "outputs/tables/cluster_ctfidf_terms.csv",
    "outputs/tables/cluster_interpretation_cards.csv",
    "outputs/tables/cluster_report_ready_summary.csv",
    "outputs/tables/cluster_entropy.csv",
    "outputs/tables/cluster_yearly_distribution.csv",
    "outputs/tables/cluster_structural_summary.csv",
    "outputs/tables/cluster_final_interpretation_cards.csv",
    "outputs/tables/cluster_final_ctfidf_terms.csv",
    "outputs/figures/umap2d_final.html",
    "outputs/figures/umap3d_final.html",
    "outputs/figures/umap2d_final.png",
    "outputs/figures/umap3d_final.png",
    "outputs/report_notes/10_final_clustering_selection.md",
    "outputs/report_notes/11_cluster_interpretation.md",
    "notebooks/tagged_corpus_analysis_execution_summary.ipynb",
]


def test_final_outputs_exist_and_are_non_empty():
    missing = [path for path in REQUIRED_OUTPUTS if not Path(path).exists()]
    assert missing == []
    empty = [path for path in REQUIRED_OUTPUTS if Path(path).stat().st_size == 0]
    assert empty == []


def test_final_dataset_schema_and_cluster_constraints():
    df = pd.read_csv("data/processed/anekdots_tagged_clustered.csv")
    required_columns = {
        "cluster_final",
        "cluster_method",
        "feature_set",
        "cluster_old_leiden",
        "cluster_leiden_tuned",
        "cluster_kmeans_best",
    }
    assert required_columns.issubset(df.columns)
    assert len(df) == 5509
    assert 8 <= df["cluster_final"].nunique() <= 25
    assert df["cluster_final"].value_counts(normalize=True).max() <= 0.40
    names = pd.read_csv("outputs/tables/cluster_interpretation_cards.csv")
    assert not names["suggested_name"].astype(str).str.fullmatch("other", case=False).any()


def test_feature_files_are_text_derived_and_row_aligned():
    df = pd.read_csv("data/processed/anekdots_tagged_clustered.csv", usecols=["id"])
    feature_paths = [
        "data/features/dense_bge_pca.npy",
        "data/features/tfidf_word_svd.npy",
        "data/features/tfidf_char_svd.npy",
        "data/features/structural_features.npy",
        "data/features/hybrid_dense_lexical.npy",
        "data/features/hybrid_dense_lexical_structural.npy",
    ]
    for path in feature_paths:
        values = np.load(path)
        assert values.shape[0] == len(df), path

    source = Path("scripts/08_feature_ablation_and_search.py").read_text(encoding="utf-8")
    body = source.split("def save_features", 1)[1].split("def balanced_score", 1)[0]
    forbidden_label_columns = ["tags_raw", "tags_norm", "macro_tags", "primary_macro_tag", "cluster_leiden", "cluster_final"]
    assert not any(column in body for column in forbidden_label_columns)


def test_executed_notebook_contains_required_outputs():
    text = Path("notebooks/tagged_corpus_analysis_execution_summary.ipynb").read_text(encoding="utf-8")
    required_fragments = [
        "Python:",
        "Torch:",
        "CUDA available:",
        "GPU:",
        "Git commit:",
        "Final cluster count:",
        "hybrid_dense_lexical_dw0.75_lw0.25",
        "generated_report_artifacts",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    assert missing == []
