# Final metrics interpretation for committee

## Main result

The main result remains the independent unsupervised Leiden clustering. Tags
were not used as clustering features and were not appended to joke text.

- ARI: 0.2768
- V-measure: 0.3871
- Pairwise multilabel F1: 0.3388

These values are moderate and should be reported honestly. They are expected
for short multi-label humor texts with noisy site-native silver labels.

## Auxiliary controls

- The level-1 hierarchical evaluation gives V-measure 0.2722 and pairwise F1 0.2477. It is not higher in this run because broad categories merge heterogeneous detailed themes and increase the number of positive pairs.
- The supervised classifier reaches micro-F1 0.7259 on the test split. This is not clustering; labels are used during training.
- The semi-supervised embedding experiment reaches holdout V-measure 0.4309 and pairwise F1 0.4141. This is a label-guided upper-bound, not independent validation.

## Recommended wording

For the thesis, use the unsupervised Leiden metrics as the main quantitative
result. Use the hierarchical, supervised, and semi-supervised tables as
supporting evidence: they show that coarse themes are easier to recover and
that the tags contain text signal, but they do not replace independent
unsupervised validation.
