# Report notes index

Use these files to update the written thesis without rerunning the pipeline:

- `00_dataset_rebuild.md` - tagged-only dataset construction, coverage, duplicates, top tags.
- `01_embeddings.md` - BGE-M3 embedding run, Colab GPU device, shape, PCA variance, truncation.
- `02_3d_umap_and_clustering.md` - Leiden clustering and 2D/3D UMAP visualization notes.
- `03_cluster_validation.md` - tag-based, pairwise multilabel, internal, and stability metrics.
- `04_weaknesses_and_improvements.md` - evidence-based limitations and copy-ready improvement notes.
- `05_audit_before_strong_metric_improvement.md` - baseline audit before the follow-up metric-improvement pass.
- `06_macro_tag_remap_strong.md` - expanded macro-tag mapping and `other` bucket reduction.
- `07_validation_metrics_exact.md` - exact external and pairwise multilabel metrics.
- `08_feature_ablation.md` - dense, lexical, structural, and hybrid text-only feature ablations.
- `09_strong_clustering_search.md` - ranked Leiden, KMeans, Agglomerative, and HDBSCAN search results.
- `10_final_clustering_selection.md` - final selected clustering configuration and before/after metrics.
- `11_cluster_interpretation.md` - c-TF-IDF terms, dominant tags, and report-ready cluster cards.

Key Phase 7 tables:

- `outputs/tables/cluster_ctfidf_terms.csv`
- `outputs/tables/cluster_interpretation_cards.csv`
- `outputs/tables/cluster_report_ready_summary.csv`
- `outputs/tables/cluster_entropy.csv`
- `outputs/tables/cluster_yearly_distribution.csv`
- `outputs/tables/cluster_structural_summary.csv`
