# Hierarchical macro-tag evaluation

This experiment keeps the final unsupervised Leiden clustering unchanged and
changes only the evaluation target.

The existing macro-tags are treated as level-2 detailed categories. They are
also mapped to broader level-1 categories in `config/macro_tag_hierarchy.yml`.
Level-1 metrics are often expected to be higher because they evaluate broader
thematic agreement: confusing two detailed categories inside the same broad
topic is less severe than confusing unrelated broad topics. This expectation is
not automatic for all metrics. In this run the broad level-1 metrics are not uniformly higher: broad categories merge many heterogeneous detailed themes, which lowers completeness and increases the number of label-positive pairs in pairwise evaluation.

## Results

| Evaluation target | Subset | ARI | AMI | NMI | Homogeneity | Completeness | V-measure | Pairwise F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Level-1 broad | all | 0.1697 | 0.2690 | 0.2722 | 0.3420 | 0.2260 | 0.2722 | 0.2477 |
| Level-2 detailed | all | 0.2768 | 0.3772 | 0.3871 | 0.3754 | 0.3996 | 0.3871 | 0.3388 |
| Level-1 broad | single-clear-label | 0.1881 | 0.2933 | 0.2968 | 0.3745 | 0.2458 | 0.2968 | 0.2650 |
| Level-2 detailed | single-clear-label | 0.3105 | 0.4091 | 0.4198 | 0.4089 | 0.4313 | 0.4198 | 0.3541 |

## Interpretation

The level-2 rows should remain the main detailed validation view, because they
compare clusters with the full macro-tag taxonomy. The level-1 rows answer a
different question: whether clusters preserve coarse thematic structure. If
level-1 numbers are higher, they can help explain broad thematic agreement. If
they are lower, as in this local run for V-measure and pairwise F1, the honest
interpretation is that the chosen broad categories are too heterogeneous for the
current cluster structure.
