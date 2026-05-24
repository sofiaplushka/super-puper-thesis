# Supervised tag-prediction baseline

This is a control experiment, not clustering. The model predicts macro-tags
from text-derived features on a train/validation/test split. It shows whether
the site-tag signal is present in the joke text when labels are allowed during
training.

Tags are not appended to joke text and are not used as input features. They are
used only as supervised targets in this separate baseline.

## Best test result

- Feature set: `hybrid_word_char_bge_pca`
- Macro-F1: 0.6665
- Micro-F1: 0.7259
- Weighted-F1: 0.7181
- Precision (micro): 0.7512
- Recall (micro): 0.7023
- Subset accuracy: 0.6723

## Interpretation

Higher supervised scores do not mean the unsupervised clustering should reach
the same values. In supervised modeling the labels directly shape the decision
boundary, while the main Leiden result remains independent of tag labels.
