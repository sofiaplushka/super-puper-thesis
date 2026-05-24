import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml


def test_macro_tag_hierarchy_covers_current_macro_tags_once():
    hierarchy = yaml.safe_load(
        Path("config/macro_tag_hierarchy.yml").read_text(encoding="utf-8")
    )
    detailed = yaml.safe_load(
        Path("config/tag_macro_categories.yml").read_text(encoding="utf-8")
    )
    assert 6 <= len(hierarchy) <= 8
    mapped = []
    for spec in hierarchy.values():
        mapped.extend(spec["level2"])
    assert set(mapped) == set(detailed)
    assert len(mapped) == len(set(mapped))


def test_hierarchical_metrics_are_present_and_aligned_with_final_level2():
    metrics = pd.read_csv("outputs/tables/hierarchical_metrics_summary.csv")
    required = {
        ("level_1_broad", "all"),
        ("level_1_broad", "single_clear_label"),
        ("level_2_detailed", "all"),
        ("level_2_detailed", "single_clear_label"),
    }
    assert set(map(tuple, metrics[["label_level", "subset"]].to_numpy())) == required
    for column in [
        "ari",
        "ami",
        "nmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "pairwise_f1",
    ]:
        assert metrics[column].between(0, 1).all()

    final = pd.read_csv("outputs/tables/final_metrics_summary.csv")
    final_all = final[(final["model"] == "final") & (final["subset"] == "all")].iloc[0]
    level2_all = metrics[
        (metrics["label_level"] == "level_2_detailed") & (metrics["subset"] == "all")
    ].iloc[0]
    assert level2_all["v_measure"] == pytest.approx(final_all["v_measure"], rel=1e-9)
    assert level2_all["pairwise_f1"] == pytest.approx(
        final_all["pairwise_f1"], rel=1e-9
    )


def test_supervised_tag_prediction_baseline_outputs_are_controls():
    path = Path("outputs/tables/supervised_tag_prediction_baseline.csv")
    assert path.exists() and path.stat().st_size > 0
    df = pd.read_csv(path)
    assert {"validation", "test"}.issubset(set(df["split"]))
    assert {
        "tfidf_word_1_2",
        "tfidf_char_wb_3_5",
        "bge_pca128",
        "hybrid_word_char_bge_pca",
    }.issubset(set(df["feature_set"]))
    for column in [
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "micro_precision",
        "micro_recall",
        "subset_accuracy",
    ]:
        assert df[column].between(0, 1).all()
    assert df[df["split"] == "test"]["micro_f1"].max() > 0.5

    note = Path(
        "outputs/report_notes/14_supervised_tag_prediction_baseline.md"
    ).read_text(encoding="utf-8")
    assert "not clustering" in note
    assert "Tags are not appended" in note


def test_semi_supervised_upper_bound_is_label_guided_and_separate():
    embedding = np.load("data/embeddings/tagged_bge_m3_finetuned.npy")
    clustered = pd.read_csv("data/processed/anekdots_tagged_clustered.csv")
    assert embedding.shape[0] == len(clustered) == 5509
    assert clustered["cluster_final"].nunique() == 20
    assert set(clustered["feature_set"]) == {"hybrid_dense_lexical_dw0.75_lw0.25"}

    manifest = json.loads(
        Path("data/embeddings/tagged_bge_m3_finetuned.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["label_guided"] is True
    assert manifest["independent_external_validation"] is False
    assert manifest["backend"] == "torch_linear_projection_contrastive"
    assert manifest["pair_stats"]["positive_pairs"] > 0
    assert manifest["pair_stats"]["hard_negative_pairs"] > 0

    metrics = pd.read_csv("outputs/tables/semi_supervised_embedding_metrics.csv")
    selected = metrics[metrics["selected"] == True]
    assert {"holdout", "full_corpus_label_guided"}.issubset(set(selected["split"]))
    assert selected["v_measure"].between(0, 1).all()
    assert selected["pairwise_f1"].between(0, 1).all()

    note = Path("outputs/report_notes/15_semi_supervised_upper_bound.md").read_text(
        encoding="utf-8"
    )
    assert "not independent external validation" in note
    assert "upper-bound" in note


def test_final_evaluation_story_and_supervisor_notebook():
    story = pd.read_csv("outputs/tables/final_evaluation_story.csv")
    assert {
        "unsupervised_leiden_final",
        "unsupervised_leiden_level1",
        "supervised_tag_classifier",
        "semi_supervised_finetuned_clustering_holdout",
    }.issubset(set(story["result_id"]))
    main = story[story["result_id"] == "unsupervised_leiden_final"].iloc[0]
    assert bool(main["independent_external_validation"]) is True
    assert bool(main["label_guided_representation"]) is False
    assert main["main_or_auxiliary"] == "main"
    assert main["v_measure"] == pytest.approx(0.38709742673182684, rel=1e-9)

    auxiliary = story[story["label_guided_representation"] == True]
    assert (auxiliary["main_or_auxiliary"] != "main").all()

    notebook_path = Path("notebooks/Sophie_анеки_кластеризация_итоговая.ipynb")
    assert notebook_path.exists()
    nb = nbformat.read(str(notebook_path), as_version=4)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    markdown_cells = [cell for cell in nb.cells if cell.cell_type == "markdown"]
    assert len(code_cells) >= 15
    assert len(markdown_cells) >= 14
    assert sum(bool(cell.get("outputs")) for cell in code_cells) >= 15
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    markdown = "\n".join(
        cell.source for cell in nb.cells if cell.cell_type == "markdown"
    )
    all_source = "\n".join(cell.source for cell in nb.cells)
    assert "Кластеризация анекдотов" in markdown
    assert "Получение эмбеддингов" in markdown
    assert "Анализ пространства эмбеддингов" in markdown
    assert "Почему не простой KMeans" in markdown
    assert "Теперь кластеризуем графом: Leiden + UMAP" in markdown
    assert "Контрольная модель с учителем" in markdown
    assert "sanity_check" in all_source
    assert "Leiden clustering + UMAP" in all_source
    assert "Saved:" in str(nb)
    assert "Рљ" not in all_source

    export = Path("outputs/tables/clustered_anekdots_for_supervisor.csv")
    assert export.exists() and export.stat().st_size > 0
    exported = pd.read_csv(export)
    assert len(exported) == 5509
    assert {"cluster_final", "umap2_x", "umap2_y", "text_clean"}.issubset(
        exported.columns
    )
