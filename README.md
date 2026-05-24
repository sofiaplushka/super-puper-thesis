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
python scripts/run_all.py --skip-existing
```

The embedding step can be run in Colab/GPU by cloning this repository in the
Colab runtime and running the same `scripts/02_compute_embeddings.py` command.
Do not commit prompt files, Colab bridge code, tokens, `.env` files, ngrok URLs,
logs, or cache directories.

