# Execution report

## Scope

This report records the execution of the seven ordered task specifications from
`/docs`:

1. `01_PREFLIGHT_AND_PROJECT_STRUCTURE.md`
2. `02_REBUILD_TAGGED_DATASET.md`
3. `03_EMBEDDINGS_LOCAL_OR_COLAB.md`
4. `04_CLUSTERING_AND_3D_VISUALIZATION.md`
5. `05_CLUSTER_VALIDATION_WITH_TAGS.md`
6. `06_WEAKNESSES_AND_IMPROVEMENTS.md`
7. `07_FINAL_RUN_AND_REPORT_GUIDE.md`

The original task prompt files were used as specifications and are not included
in the committed changes.

## Commands run

```bash
pytest -q
python scripts/01_build_tagged_dataset.py --start-year 1996 --start-month 1 --end-year 2026 --end-month 3 --max-per-month 30 --seed 42 --sleep 0.2 --output data/processed/anekdots_tagged.csv
python scripts/02_compute_embeddings.py --input data/processed/anekdots_tagged.csv --text-column text --id-column id --model BAAI/bge-m3 --batch-size 32 --mode local --output-dir data/embeddings --seed 42
python scripts/03_cluster_and_visualize.py --dataset data/processed/anekdots_tagged.csv --embeddings data/embeddings/tagged_bge_m3.npy --pca data/embeddings/tagged_pca128.npy --k 30 --resolution 1.0 --seed 42
python scripts/04_validate_clusters_with_tags.py --clustered data/processed/anekdots_tagged_clustered.csv --embeddings data/embeddings/tagged_bge_m3.npy --pca data/embeddings/tagged_pca128.npy --tag-map config/tag_macro_categories.yml --seed 42
python scripts/05_analyze_practical_weaknesses.py --dataset data/processed/anekdots_tagged_clustered.csv --validation-dir outputs/tables --seed 42
python scripts/06_remap_macro_tags.py
python scripts/07_compute_exact_metrics.py
python scripts/08_feature_ablation_and_search.py
python scripts/09_select_final_and_interpret.py
python scripts/11_compute_final_internal_metrics.py
python scripts/12_macro_tag_mapping_audit.py
python scripts/10_build_execution_summary_notebook.py
```

Google Colab was used only for GPU embedding generation on a Tesla T4 through
the temporary bridge supplied by the user. Colab bridge code, bridge tokens,
ngrok URLs, logs, and secrets were not copied into this repository. The executed
summary notebook referenced below is a local reproducibility summary, not a
claim that the notebook itself was executed in Colab.

## Results by prompt

### 01 Preflight and structure

Created reproducible project structure:

- `config/`
- `data/raw/`
- `data/processed/`
- `data/embeddings/`
- `outputs/figures/`
- `outputs/tables/`
- `outputs/report_notes/`
- `scripts/`
- `src/thesis_pipeline/`
- `tests/`
- `notebooks/`

Added `.gitignore`, `requirements.txt`, `pytest.ini`, README, package modules,
and starter tests. Branch name is `main`, which does not contain the forbidden
word.

### 02 Tagged dataset rebuild

Generated:

- `data/raw/anekdot_tagged_candidates.csv`
- `data/processed/anekdots_tagged.csv`
- `data/processed/tag_dictionary.csv`
- `data/processed/tag_macro_mapping.csv`
- `outputs/tables/dataset_month_coverage.csv`
- `outputs/tables/dataset_year_coverage.csv`
- `outputs/tables/top_tags.csv`
- `outputs/tables/top_macro_tags.csv`
- `outputs/tables/duplicate_groups.csv`
- `outputs/report_notes/00_dataset_rebuild.md`

Final dataset rows: **5,509**. Every row has `tag_count >= 1`; max selected rows
per month is 30; duplicate ids are absent.

### 03 Embeddings

Generated:

- `data/embeddings/tagged_bge_m3.npy`
- `data/embeddings/tagged_pca128.npy`
- `data/embeddings/tagged_ids.npy`
- `data/embeddings/tagged_text_hashes.csv`
- `data/embeddings/tagged_embeddings_manifest.json`
- `outputs/report_notes/01_embeddings.md`

Embedding shape: **5,509 x 1,024**. PCA shape: **5,509 x 128**. Device:
`cuda:0` on Colab T4. Runtime: **124.072 seconds**. Token truncation count: **0**.

### 04 Clustering and 3D visualization

Generated:

- `data/processed/anekdots_tagged_clustered.csv`
- `data/embeddings/tagged_umap2d.npy`
- `data/embeddings/tagged_umap3d.npy`
- `outputs/figures/umap2d_leiden.html`
- `outputs/figures/umap3d_leiden.html`
- `outputs/figures/pca3d_baseline.html`
- `outputs/tables/cluster_sizes.csv`
- `outputs/tables/cluster_summary.csv`
- `outputs/tables/cluster_central_examples.csv`
- `outputs/tables/cluster_borderline_examples.csv`
- `outputs/report_notes/02_3d_umap_and_clustering.md`

Leiden clusters: **12**. Modularity: **0.5358**. Clustering was run in PCA
embedding space; UMAP was used only for visualization.

### 05 Cluster validation

Generated tag distributions, cluster-tag matrices, heatmap, external metrics,
pairwise multilabel metrics, internal metrics, and stability tables.

Key values:

- Single-label ARI: **0.1332**
- Single-label AMI: **0.2625**
- Single-label V-measure: **0.2658**
- Pairwise multilabel F1: **0.2589**
- Silhouette cosine: **0.0366**

### 06 Weakness diagnostics

Generated:

- `outputs/tables/weakness_dataset_bias.csv`
- `outputs/tables/weakness_duplicate_examples.csv`
- `outputs/tables/weakness_length_by_cluster.csv`
- `outputs/tables/weakness_length_by_macro_tag.csv`
- `outputs/tables/cluster_interpretability_summary.csv`
- `outputs/tables/structural_features_by_cluster.csv`
- `outputs/tables/structural_features_by_macro_tag.csv`
- `outputs/tables/weakness_robustness_summary.csv`
- `outputs/figures/year_coverage.html`
- `outputs/figures/month_coverage.html`
- `outputs/figures/text_length_distribution.html`
- `outputs/report_notes/04_weaknesses_and_improvements.md`

### 07 Final run and guide

Added:

- `scripts/run_all.py`
- `docs/report_update_guide.md`
- `outputs/report_notes/README.md`
- `notebooks/tagged_corpus_analysis.ipynb`

### Strong metric improvement extension

After the initial seven-stage run, the tag-to-macro mapping was expanded so all
158 observed raw tags map to explicit macro-categories. The previous `other`
bucket had 1,830 tag occurrences; after remapping it has **0**. Tags remain
validation and interpretation data only; clustering features are derived from
joke text, text embeddings, TF-IDF/SVD text features, and non-label structural
text properties.

Generated:

- `outputs/report_notes/05_audit_before_strong_metric_improvement.md`
- `outputs/report_notes/06_macro_tag_remap_strong.md`
- `outputs/report_notes/07_validation_metrics_exact.md`
- `outputs/report_notes/08_feature_ablation.md`
- `outputs/report_notes/09_strong_clustering_search.md`
- `outputs/report_notes/10_final_clustering_selection.md`
- `outputs/report_notes/11_cluster_interpretation.md`
- `outputs/tables/feature_ablation_metrics.csv`
- `outputs/tables/clustering_search_all_runs.csv`
- `outputs/tables/final_clustering_selection.csv`
- `outputs/tables/final_metrics_summary.csv`
- `outputs/tables/cluster_ctfidf_terms.csv`
- `outputs/tables/cluster_interpretation_cards.csv`
- `outputs/tables/cluster_report_ready_summary.csv`
- `outputs/tables/cluster_final_interpretation_cards.csv`
- `outputs/tables/final_internal_cluster_metrics.csv`
- `outputs/tables/macro_tag_mapping_audit.csv`
- `outputs/figures/umap2d_final.html`
- `outputs/figures/umap3d_final.html`
- `outputs/figures/umap2d_final.png`
- `outputs/figures/umap3d_final.png`
- `notebooks/tagged_corpus_analysis_execution_summary.ipynb`
- `outputs/report_notes/12_macro_tag_mapping_audit.md`

Final selected configuration:

- Method: **Leiden**
- Feature set: **hybrid BGE/PCA + lexical SVD**, `dense_weight=0.75`, `lexical_weight=0.25`
- Parameters: `k=75`, `resolution=2.0`, `seed=7`
- Clusters: **20**
- Largest cluster share: **0.1291**

Before/after metrics against the remapped old Leiden baseline:

- Excluding-other ARI: **0.1967 -> 0.2768** (`+0.0802`)
- Excluding-other V-measure: **0.3109 -> 0.3871** (`+0.0762`)
- Exact pairwise multilabel F1: **0.2846 -> 0.3388** (`+0.0542`)
- Exact pairwise precision: **0.2297 -> 0.3519** (`+0.1221`)
- Largest cluster share: **0.2046 -> 0.1291**

The strict stretch thresholds from the follow-up goal were not all reached:
ARI did not improve by `+0.15`, V-measure did not reach `0.40` under the
8-25-cluster constraint, and pairwise F1 remained below `0.35` on the full
multi-label set. The single-clear-label subset reached pairwise F1 **0.3541**
and V-measure **0.4198**. These results are reported as-is rather than inflated.

Final internal metrics are stored in
`outputs/tables/final_internal_cluster_metrics.csv` and mirrored to
`outputs/tables/internal_cluster_metrics.csv`; the initial 12-cluster internal
metrics were archived as `outputs/tables/internal_cluster_metrics_initial_leiden.csv`.
The execution notebook is a local nbclient reproducibility summary based on
saved repository artifacts. It is not described as a real Colab execution
notebook.

## Rerun locally

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_all.py --skip-existing
```

If embeddings need to be recomputed locally, run:

```bash
python scripts/02_compute_embeddings.py --mode local --batch-size 32
```

## Rerun in Colab

Clone the repository in a GPU runtime and run:

```bash
pip install -r requirements.txt
python scripts/02_compute_embeddings.py --mode local --batch-size 32
```

Then download or commit the generated `data/embeddings/*` and
`outputs/report_notes/01_embeddings.md` artifacts.

## Unresolved issues

No blocking code issue is known at this point. The main methodological
limitations are that the corpus is tagged-only, tags are silver labels rather
than expert gold labels, the `other=0` result comes from an exhaustive tag-level
taxonomy rather than expert row labeling, and the strongest honest final
configuration improves the metrics substantially but does not meet every stretch
target from the metric-improvement specification.
