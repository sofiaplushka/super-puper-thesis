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
```

Embedding computation was executed in Google Colab on a Tesla T4 through the
temporary bridge supplied by the user. Colab bridge code, bridge tokens, ngrok
URLs, logs, and secrets were not copied into this repository.

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

No blocking issue is known at this point. The main methodological limitation is
that the corpus is tagged-only and tags are silver labels, not expert gold
labels.

