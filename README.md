# Tagged anekdot.ru thesis pipeline

This repository keeps the old thesis artifacts as references, but the current
reproducible pipeline uses the tagged-only dataset:

```text
data/processed/anekdots_tagged.csv
```

The main command sequence is:

```bash
python scripts/01_build_tagged_dataset.py
python scripts/02_compute_embeddings.py --mode local
python scripts/03_cluster_and_visualize.py
python scripts/04_validate_clusters_with_tags.py
python scripts/05_analyze_practical_weaknesses.py
python scripts/06_remap_macro_tags.py
python scripts/07_compute_exact_metrics.py
python scripts/08_feature_ablation_and_search.py
python scripts/09_select_final_and_interpret.py
python scripts/11_compute_final_internal_metrics.py
python scripts/12_macro_tag_mapping_audit.py
python scripts/10_build_execution_summary_notebook.py
python scripts/run_all.py --skip-existing
```

The embedding step can be run in Colab/GPU by cloning this repository in the
Colab runtime and running the same `scripts/02_compute_embeddings.py` command.
The final clustering result is Leiden over `hybrid_dense_lexical_dw0.75_lw0.25`
with `k=75`, `resolution=2.0`, `seed=7`, 20 clusters, and report artifacts under
`outputs/`.
Do not commit prompt files, Colab bridge code, tokens, `.env` files, ngrok URLs,
logs, or cache directories.
