from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def first_row(df: pd.DataFrame, **filters) -> dict[str, object]:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key].eq(value)
    part = df.loc[mask]
    if part.empty:
        raise ValueError(f"No row for filters: {filters}")
    return part.iloc[0].to_dict()


def write_note(story: pd.DataFrame, output: str | Path) -> None:
    main = story[story["result_id"].eq("unsupervised_leiden_final")].iloc[0]
    hierarchical = story[story["result_id"].eq("unsupervised_leiden_level1")].iloc[0]
    supervised = story[story["result_id"].eq("supervised_tag_classifier")].iloc[0]
    semi = story[story["result_id"].eq("semi_supervised_finetuned_clustering_holdout")].iloc[0]
    level1_comment = (
        f"The level-1 hierarchical evaluation gives V-measure {hierarchical['v_measure']:.4f} and pairwise F1 {hierarchical['pairwise_f1']:.4f}. It is a broader target and may be higher when clusters mainly confuse neighboring detailed themes."
        if hierarchical["v_measure"] >= main["v_measure"]
        else f"The level-1 hierarchical evaluation gives V-measure {hierarchical['v_measure']:.4f} and pairwise F1 {hierarchical['pairwise_f1']:.4f}. It is not higher in this run because broad categories merge heterogeneous detailed themes and increase the number of positive pairs."
    )
    lines = [
        "# Final metrics interpretation for committee",
        "",
        "## Main result",
        "",
        "The main result remains the independent unsupervised Leiden clustering. Tags",
        "were not used as clustering features and were not appended to joke text.",
        "",
        f"- ARI: {main['ari']:.4f}",
        f"- V-measure: {main['v_measure']:.4f}",
        f"- Pairwise multilabel F1: {main['pairwise_f1']:.4f}",
        "",
        "These values are moderate and should be reported honestly. They are expected",
        "for short multi-label humor texts with noisy site-native silver labels.",
        "",
        "## Auxiliary controls",
        "",
        f"- {level1_comment}",
        f"- The supervised classifier reaches micro-F1 {supervised['micro_f1']:.4f} on the test split. This is not clustering; labels are used during training.",
        f"- The semi-supervised embedding experiment reaches holdout V-measure {semi['v_measure']:.4f} and pairwise F1 {semi['pairwise_f1']:.4f}. This is a label-guided upper-bound, not independent validation.",
        "",
        "## Recommended wording",
        "",
        "For the thesis, use the unsupervised Leiden metrics as the main quantitative",
        "result. Use the hierarchical, supervised, and semi-supervised tables as",
        "supporting evidence: they show that coarse themes are easier to recover and",
        "that the tags contain text signal, but they do not replace independent",
        "unsupervised validation.",
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/tables/final_evaluation_story.csv")
    parser.add_argument("--note", default="outputs/report_notes/16_final_metrics_interpretation_for_committee.md")
    args = parser.parse_args()

    final = pd.read_csv("outputs/tables/final_metrics_summary.csv")
    hierarchical = pd.read_csv("outputs/tables/hierarchical_metrics_summary.csv")
    supervised = pd.read_csv("outputs/tables/supervised_tag_prediction_baseline.csv")
    semi = pd.read_csv("outputs/tables/semi_supervised_embedding_metrics.csv")

    final_all = first_row(final, model="final", subset="all")
    level1 = first_row(hierarchical, label_level="level_1_broad", subset="all")
    level2 = first_row(hierarchical, label_level="level_2_detailed", subset="all")
    supervised_best = supervised[supervised["split"].eq("test")].sort_values("micro_f1", ascending=False).iloc[0].to_dict()
    semi_holdout = semi[(semi["split"].eq("holdout")) & (semi["selected"].eq(True))].iloc[0].to_dict()
    semi_full = semi[(semi["split"].eq("full_corpus_label_guided")) & (semi["selected"].eq(True))].iloc[0].to_dict()

    rows = [
        {
            "result_id": "unsupervised_leiden_final",
            "display_name": "Unsupervised Leiden final",
            "method_type": "unsupervised_clustering",
            "evaluation_scope": "full corpus, level-2 detailed macro-tags",
            "main_or_auxiliary": "main",
            "independent_external_validation": True,
            "label_guided_representation": False,
            "ari": final_all["ari"],
            "ami": final_all["ami"],
            "nmi": final_all["nmi"],
            "homogeneity": final_all["homogeneity"],
            "completeness": final_all["completeness"],
            "v_measure": final_all["v_measure"],
            "pairwise_f1": final_all["pairwise_f1"],
            "macro_f1": np.nan,
            "micro_f1": np.nan,
            "weighted_f1": np.nan,
            "precision": final_all["pairwise_precision"],
            "recall": final_all["pairwise_recall"],
            "subset_accuracy": np.nan,
            "note": "Primary result to report.",
        },
        {
            "result_id": "unsupervised_leiden_level1",
            "display_name": "Unsupervised Leiden evaluated at level-1",
            "method_type": "hierarchical_evaluation",
            "evaluation_scope": "full corpus, broad level-1 macro-tags",
            "main_or_auxiliary": "auxiliary",
            "independent_external_validation": True,
            "label_guided_representation": False,
            "ari": level1["ari"],
            "ami": level1["ami"],
            "nmi": level1["nmi"],
            "homogeneity": level1["homogeneity"],
            "completeness": level1["completeness"],
            "v_measure": level1["v_measure"],
            "pairwise_f1": level1["pairwise_f1"],
            "macro_f1": np.nan,
            "micro_f1": np.nan,
            "weighted_f1": np.nan,
            "precision": level1["pairwise_precision"],
            "recall": level1["pairwise_recall"],
            "subset_accuracy": np.nan,
            "note": "Same clustering, broader evaluation target; not automatically higher.",
        },
        {
            "result_id": "unsupervised_leiden_level2",
            "display_name": "Unsupervised Leiden evaluated at level-2",
            "method_type": "hierarchical_evaluation",
            "evaluation_scope": "full corpus, detailed level-2 macro-tags",
            "main_or_auxiliary": "auxiliary",
            "independent_external_validation": True,
            "label_guided_representation": False,
            "ari": level2["ari"],
            "ami": level2["ami"],
            "nmi": level2["nmi"],
            "homogeneity": level2["homogeneity"],
            "completeness": level2["completeness"],
            "v_measure": level2["v_measure"],
            "pairwise_f1": level2["pairwise_f1"],
            "macro_f1": np.nan,
            "micro_f1": np.nan,
            "weighted_f1": np.nan,
            "precision": level2["pairwise_precision"],
            "recall": level2["pairwise_recall"],
            "subset_accuracy": np.nan,
            "note": "Should match detailed final validation except for hierarchy plumbing.",
        },
        {
            "result_id": "supervised_tag_classifier",
            "display_name": "Supervised tag classifier",
            "method_type": "supervised_multilabel_classification",
            "evaluation_scope": "held-out test split",
            "main_or_auxiliary": "auxiliary_control",
            "independent_external_validation": False,
            "label_guided_representation": True,
            "ari": np.nan,
            "ami": np.nan,
            "nmi": np.nan,
            "homogeneity": np.nan,
            "completeness": np.nan,
            "v_measure": np.nan,
            "pairwise_f1": np.nan,
            "macro_f1": supervised_best["macro_f1"],
            "micro_f1": supervised_best["micro_f1"],
            "weighted_f1": supervised_best["weighted_f1"],
            "precision": supervised_best["micro_precision"],
            "recall": supervised_best["micro_recall"],
            "subset_accuracy": supervised_best["subset_accuracy"],
            "note": f"Best test feature set: {supervised_best['feature_set']}. Not clustering.",
        },
        {
            "result_id": "semi_supervised_finetuned_clustering_holdout",
            "display_name": "Semi-supervised fine-tuned clustering holdout",
            "method_type": "semi_supervised_clustering_upper_bound",
            "evaluation_scope": "holdout split",
            "main_or_auxiliary": "auxiliary_upper_bound",
            "independent_external_validation": False,
            "label_guided_representation": True,
            "ari": semi_holdout["ari"],
            "ami": semi_holdout["ami"],
            "nmi": semi_holdout["nmi"],
            "homogeneity": semi_holdout["homogeneity"],
            "completeness": semi_holdout["completeness"],
            "v_measure": semi_holdout["v_measure"],
            "pairwise_f1": semi_holdout["pairwise_f1"],
            "macro_f1": np.nan,
            "micro_f1": np.nan,
            "weighted_f1": np.nan,
            "precision": semi_holdout["pairwise_precision"],
            "recall": semi_holdout["pairwise_recall"],
            "subset_accuracy": np.nan,
            "note": "Validation labels selected the representation/parameters; not independent.",
        },
        {
            "result_id": "semi_supervised_finetuned_clustering_full",
            "display_name": "Semi-supervised fine-tuned clustering full corpus",
            "method_type": "semi_supervised_clustering_upper_bound",
            "evaluation_scope": "full corpus, label-guided",
            "main_or_auxiliary": "auxiliary_upper_bound",
            "independent_external_validation": False,
            "label_guided_representation": True,
            "ari": semi_full["ari"],
            "ami": semi_full["ami"],
            "nmi": semi_full["nmi"],
            "homogeneity": semi_full["homogeneity"],
            "completeness": semi_full["completeness"],
            "v_measure": semi_full["v_measure"],
            "pairwise_f1": semi_full["pairwise_f1"],
            "macro_f1": np.nan,
            "micro_f1": np.nan,
            "weighted_f1": np.nan,
            "precision": semi_full["pairwise_precision"],
            "recall": semi_full["pairwise_recall"],
            "subset_accuracy": np.nan,
            "note": "Full-corpus label-guided number for context only.",
        },
    ]
    story = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    story.to_csv(args.output, index=False, encoding="utf-8")
    write_note(story, args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
