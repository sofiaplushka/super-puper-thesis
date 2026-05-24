# 3D UMAP and Leiden clustering

Dataset: `data/processed/anekdots_tagged.csv`
Embeddings: `data/embeddings/tagged_bge_m3.npy`
PCA input: `data/embeddings/tagged_pca128.npy`
kNN k: `30`
Leiden resolution: `1.0`
Seed: `42`
Leiden fallback used: `False`
Modularity: `0.5358356252247378`
Number of clusters: **12**
Largest/median cluster share ratio: `2.385`

Clustering is performed in BGE-M3/PCA embedding space. UMAP-2D and UMAP-3D are visualization
layers only and are not used as clustering inputs.

## Cluster sizes
|   cluster_leiden |   size |     share |
|-----------------:|-------:|----------:|
|                0 |   1127 | 0.204574  |
|                1 |    746 | 0.135415  |
|                2 |    722 | 0.131058  |
|                3 |    529 | 0.0960247 |
|                4 |    515 | 0.0934834 |
|                5 |    504 | 0.0914867 |
|                6 |    441 | 0.0800508 |
|                7 |    230 | 0.0417499 |
|                8 |    217 | 0.0393901 |
|                9 |    209 | 0.0379379 |
|               10 |    176 | 0.0319477 |
|               11 |     93 | 0.0168815 |

## Copy-ready practical section notes
The updated practical pipeline uses only jokes that have at least one anekdot.ru tag.
Embeddings are computed before any dimensional visualization, and Leiden clustering is applied
to the high-dimensional embedding/PCA representation rather than to UMAP coordinates.
The 3D UMAP figure is therefore a visual projection of the same Leiden labels used in the 2D plot.
This distinction is important because UMAP can change local visual geometry while preserving only
an approximate neighborhood structure.
Cluster summaries, central examples, and borderline examples were exported to CSV so the written
thesis can discuss both strong and weak cluster interpretations without rerunning the notebook.