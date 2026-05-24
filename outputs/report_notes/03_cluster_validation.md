# Cluster validation with site tags

Tags are external silver labels from anekdot.ru, not manual gold labels. Multi-label jokes
are handled explicitly through cluster-tag matrices and pairwise multilabel metrics.

## Single-label external metrics
|   adjusted_rand_index |   adjusted_mutual_info |   homogeneity |   completeness |   v_measure |   rows |
|----------------------:|-----------------------:|--------------:|---------------:|------------:|-------:|
|              0.133207 |               0.262534 |        0.2854 |       0.248774 |    0.265831 |   5243 |

## Primary macro approximation metrics
|   adjusted_rand_index |   adjusted_mutual_info |   homogeneity |   completeness |   v_measure |   rows | label_source                                    |
|----------------------:|-----------------------:|--------------:|---------------:|------------:|-------:|:------------------------------------------------|
|              0.139706 |               0.257268 |      0.278402 |       0.244614 |    0.260417 |   5509 | primary_macro_tag_all_rows_weaker_approximation |

## Pairwise multilabel metrics
| exact   |   total_possible_pairs |   pairs_evaluated |   true_positive |   false_positive |   false_negative |   true_negative |   precision |   recall |       f1 |
|:--------|-----------------------:|------------------:|----------------:|-----------------:|-----------------:|----------------:|------------:|---------:|---------:|
| False   |               15171786 |           1000000 |           38571 |            77919 |           142941 |          740569 |     0.33111 | 0.212498 | 0.258864 |

## Internal metrics
| metric                |       value | space            |
|:----------------------|------------:|:-----------------|
| silhouette_cosine     |   0.0365702 | embedding_or_pca |
| calinski_harabasz     |  58.279     | pca              |
| davies_bouldin        |   4.80071   | pca              |
| modularity            | nan         | knn_graph        |
| cluster_count         |  12         | labels           |
| largest_cluster_share |   0.204574  | labels           |

## Interpretation
High homogeneity would mean that clusters rarely mix macro tags, while high completeness would
mean that a macro tag is concentrated inside one cluster. For humor data both values are expected
to be imperfect because tags are noisy, broad, and multi-label. Pairwise F1 is a stricter view:
it checks whether pairs placed in the same cluster also share at least one macro tag.

## Copy-ready report paragraphs
The updated validation treats anekdot.ru tags as silver labels. This avoids manual row labeling
but also means the evaluation measures agreement with the site's topic taxonomy rather than
absolute semantic truth.
Because jokes may have several tags, the report includes both single-label metrics on a filtered
subset and pairwise multilabel precision/recall/F1 on the full dataset.
Internal metrics are computed in embedding/PCA space and never on UMAP coordinates.
Stability is checked by varying kNN graph size, Leiden resolution, and random seeds.

## Suggested captions
Table: External cluster quality metrics against single macro tags.
Table: Pairwise multilabel cluster agreement using shared macro tags.
Figure: Row-normalized cluster by macro-tag heatmap.