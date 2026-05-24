# Semi-supervised embedding upper-bound

This experiment is intentionally separate from the main unsupervised Leiden
result. Macro-tags are used on the train and validation splits to shape the
representation and select clustering parameters. Therefore the resulting
metrics are not independent external validation.

The local run trained a lightweight contrastive projection head over the saved
BGE/PCA embeddings. It created positive pairs from jokes sharing at least one
macro-tag, random negative pairs from jokes with disjoint macro-tags, and hard
negative pairs from nearest neighbors with disjoint macro-tags.

## Pair sampling

- Positive pairs: 6572
- Negative pairs: 9235
- Hard negative pairs: 2630
- Total pairs: 15807

## Selected validation configuration

- Params: `k=30;resolution=2.0;seed=42`
- Validation selection score: 0.4642

## Holdout and full-corpus metrics

- Holdout ARI: 0.3501
- Holdout V-measure: 0.4309
- Holdout pairwise F1: 0.4141
- Full-corpus label-guided ARI: 0.4030
- Full-corpus label-guided V-measure: 0.4341
- Full-corpus label-guided pairwise F1: 0.4634

## Methodological warning

These values should be described as an upper-bound or label-guided control.
They must not replace the main unsupervised clustering metrics because the
tag labels influenced the learned representation and validation selection.

The current local environment did not have `sentence-transformers`
installed at run time, so the committed artifact uses the lightweight
projection-head fallback rather than a full SentenceTransformer encoder
fine-tune. The distinction is recorded in the manifest.

See also `outputs/tables/semi_supervised_vs_unsupervised.csv`.
