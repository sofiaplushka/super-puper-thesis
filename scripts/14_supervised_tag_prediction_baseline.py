from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.linear_model import SGDClassifier

from thesis_pipeline.tag_mapping import parse_json_list


def multilabel_sets(series: pd.Series) -> list[list[str]]:
    return [sorted(set(parse_json_list(value))) for value in series]


def split_indices(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    stratify = df["primary_macro_tag"].astype(str)
    try:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=0.30,
            random_state=seed,
            stratify=stratify,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=seed,
            stratify=stratify.iloc[temp_idx],
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def force_one_label(scores: np.ndarray) -> np.ndarray:
    pred = scores >= 0.0
    empty = np.where(pred.sum(axis=1) == 0)[0]
    if len(empty):
        pred[empty, np.argmax(scores[empty], axis=1)] = True
    return pred.astype(int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def classifier(seed: int) -> OneVsRestClassifier:
    base = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1500,
        tol=1e-3,
        random_state=seed,
        n_jobs=1,
    )
    return OneVsRestClassifier(base, n_jobs=1)


def run_feature_set(
    name: str,
    x_train,
    x_val,
    x_test,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    train_rows: int,
    val_rows: int,
    test_rows: int,
) -> list[dict[str, object]]:
    model = classifier(seed)
    model.fit(x_train, y_train)
    rows = []
    for split, x, y in [("validation", x_val, y_val), ("test", x_test, y_test)]:
        scores = model.decision_function(x)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        pred = force_one_label(np.asarray(scores))
        row: dict[str, object] = {
            "experiment": "supervised_tag_prediction_baseline",
            "feature_set": name,
            "split": split,
            "rows": int(len(y)),
            "label_count": int(y.shape[1]),
            "train_rows": int(train_rows),
            "validation_rows": int(val_rows),
            "test_rows": int(test_rows),
            "method": "OneVsRestClassifier(SGDClassifier logistic)",
            "feature_source": "text-derived only; no tag strings appended to text",
        }
        row.update(evaluate(y, pred))
        rows.append(row)
    return rows


def write_note(results: pd.DataFrame, output: str | Path) -> None:
    test = results[results["split"].eq("test")].sort_values("micro_f1", ascending=False)
    best = test.iloc[0].to_dict()
    lines = [
        "# Supervised tag-prediction baseline",
        "",
        "This is a control experiment, not clustering. The model predicts macro-tags",
        "from text-derived features on a train/validation/test split. It shows whether",
        "the site-tag signal is present in the joke text when labels are allowed during",
        "training.",
        "",
        "Tags are not appended to joke text and are not used as input features. They are",
        "used only as supervised targets in this separate baseline.",
        "",
        "## Best test result",
        "",
        f"- Feature set: `{best['feature_set']}`",
        f"- Macro-F1: {best['macro_f1']:.4f}",
        f"- Micro-F1: {best['micro_f1']:.4f}",
        f"- Weighted-F1: {best['weighted_f1']:.4f}",
        f"- Precision (micro): {best['micro_precision']:.4f}",
        f"- Recall (micro): {best['micro_recall']:.4f}",
        f"- Subset accuracy: {best['subset_accuracy']:.4f}",
        "",
        "## Interpretation",
        "",
        "Higher supervised scores do not mean the unsupervised clustering should reach",
        "the same values. In supervised modeling the labels directly shape the decision",
        "boundary, while the main Leiden result remains independent of tag labels.",
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/anekdots_tagged.csv")
    parser.add_argument("--pca", default="data/embeddings/tagged_pca128.npy")
    parser.add_argument("--output", default="outputs/tables/supervised_tag_prediction_baseline.csv")
    parser.add_argument("--note", default="outputs/report_notes/14_supervised_tag_prediction_baseline.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    texts = df["text_clean" if "text_clean" in df.columns else "text"].fillna("").astype(str)
    labels = multilabel_sets(df["macro_tags"])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    train_idx, val_idx, test_idx = split_indices(df, args.seed)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
        lowercase=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
        lowercase=True,
    )
    word_train = word.fit_transform(texts.iloc[train_idx])
    word_val = word.transform(texts.iloc[val_idx])
    word_test = word.transform(texts.iloc[test_idx])
    char_train = char.fit_transform(texts.iloc[train_idx])
    char_val = char.transform(texts.iloc[val_idx])
    char_test = char.transform(texts.iloc[test_idx])
    dense = normalize(np.load(args.pca), norm="l2")
    dense_sparse = sparse.csr_matrix(dense)

    feature_sets = {
        "tfidf_word_1_2": (word_train, word_val, word_test),
        "tfidf_char_wb_3_5": (char_train, char_val, char_test),
        "bge_pca128": (dense_sparse[train_idx], dense_sparse[val_idx], dense_sparse[test_idx]),
        "hybrid_word_char_bge_pca": (
            sparse.hstack([word_train, char_train, dense_sparse[train_idx]], format="csr"),
            sparse.hstack([word_val, char_val, dense_sparse[val_idx]], format="csr"),
            sparse.hstack([word_test, char_test, dense_sparse[test_idx]], format="csr"),
        ),
    }

    rows: list[dict[str, object]] = []
    for name, matrices in feature_sets.items():
        rows.extend(
            run_feature_set(
                name,
                matrices[0],
                matrices[1],
                matrices[2],
                y_train,
                y_val,
                y_test,
                args.seed,
                len(train_idx),
                len(val_idx),
                len(test_idx),
            )
        )
    result = pd.DataFrame(rows).sort_values(["split", "micro_f1"], ascending=[True, False])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")
    write_note(result, args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
