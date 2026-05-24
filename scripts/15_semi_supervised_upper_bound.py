from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from thesis_pipeline.clustering import leiden_cluster
from thesis_pipeline.strong_metrics import metric_row
from thesis_pipeline.tag_mapping import parse_json_list


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    holdout_idx: np.ndarray


def macro_sets(series: pd.Series) -> list[set[str]]:
    return [set(parse_json_list(value)) for value in series]


def primary_labels(tag_sets: list[set[str]]) -> list[str]:
    return [sorted(tags)[0] if tags else "other" for tags in tag_sets]


def split_indices(tag_sets: list[set[str]], seed: int) -> Split:
    idx = np.arange(len(tag_sets))
    primary = np.asarray(primary_labels(tag_sets))
    try:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=0.40,
            random_state=seed,
            stratify=primary,
        )
        val_idx, holdout_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=seed,
            stratify=primary[temp_idx],
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=seed)
        val_idx, holdout_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=seed
        )
    return Split(np.asarray(train_idx), np.asarray(val_idx), np.asarray(holdout_idx))


def disjoint(a: set[str], b: set[str]) -> bool:
    return not bool(a.intersection(b))


def sample_pairs(
    features: np.ndarray,
    tag_sets: list[set[str]],
    train_idx: np.ndarray,
    seed: int,
    positives_per_anchor: int,
    random_negatives_per_anchor: int,
    hard_negatives_per_anchor: int,
) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    train_set = set(int(i) for i in train_idx)
    tag_to_indices: dict[str, list[int]] = {}
    for i in train_idx:
        for tag in tag_sets[int(i)]:
            tag_to_indices.setdefault(tag, []).append(int(i))

    rows: list[tuple[int, int, int, str]] = []
    for i in train_idx:
        i = int(i)
        positive_pool = sorted(
            {
                candidate
                for tag in tag_sets[i]
                for candidate in tag_to_indices.get(tag, [])
                if candidate != i and candidate in train_set
            }
        )
        if positive_pool:
            chosen = rng.choice(
                positive_pool,
                size=min(positives_per_anchor, len(positive_pool)),
                replace=False,
            )
            for j in chosen:
                rows.append((i, int(j), 1, "positive_shared_macro_tag"))

        tries = 0
        added = 0
        while added < random_negatives_per_anchor and tries < 200:
            j = int(rng.choice(train_idx))
            tries += 1
            if j == i or not disjoint(tag_sets[i], tag_sets[j]):
                continue
            rows.append((i, j, 0, "negative_disjoint_macro_tags"))
            added += 1

    nn = NearestNeighbors(n_neighbors=min(31, len(train_idx)), metric="cosine")
    nn.fit(features[train_idx])
    _, neigh = nn.kneighbors(features[train_idx])
    for local_i, neighbors in enumerate(neigh):
        i = int(train_idx[local_i])
        added = 0
        for local_j in neighbors[1:]:
            j = int(train_idx[int(local_j)])
            if j == i or not disjoint(tag_sets[i], tag_sets[j]):
                continue
            rows.append((i, j, 0, "hard_negative_nearest_disjoint"))
            added += 1
            if added >= hard_negatives_per_anchor:
                break

    dedup: dict[tuple[int, int, int], str] = {}
    for a, b, y, source in rows:
        key = (min(a, b), max(a, b), y)
        dedup.setdefault(key, source)
    pair_rows = [(a, b, y) for (a, b, y), _ in dedup.items()]
    stats = {
        "positive_pairs": int(sum(1 for _, _, y in pair_rows if y == 1)),
        "negative_pairs": int(sum(1 for _, _, y in pair_rows if y == 0)),
        "hard_negative_pairs": int(
            sum(
                1
                for source in dedup.values()
                if source == "hard_negative_nearest_disjoint"
            )
        ),
        "total_pairs": int(len(pair_rows)),
    }
    return np.asarray(pair_rows, dtype=np.int64), stats


class Projection(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.linear(x))
        return torch.nn.functional.normalize(z, dim=1)


def train_projection(
    features: np.ndarray,
    pairs: np.ndarray,
    seed: int,
    output_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    model = Projection(features.shape[1], output_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(pair_tensor))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch = pair_tensor[order[start : start + batch_size]]
            z_a = model(x[batch[:, 0]])
            z_b = model(x[batch[:, 1]])
            y = batch[:, 2].float()
            sim = (z_a * z_b).sum(dim=1)
            pos_loss = y * (1.0 - sim).pow(2)
            neg_loss = (1.0 - y) * torch.relu(sim - margin).pow(2)
            loss = (pos_loss + neg_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    with torch.no_grad():
        embeddings = model(x).cpu().numpy().astype("float32", copy=False)
    return embeddings, {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "epochs": float(epochs),
    }


def evaluate_subset(
    df: pd.DataFrame,
    labels: np.ndarray,
    mask_idx: np.ndarray,
    split_name: str,
    run: dict[str, object],
) -> dict[str, object]:
    mask = np.zeros(len(df), dtype=bool)
    mask[mask_idx] = True
    row = metric_row(df, labels, split_name, mask)
    row.update(run)
    row["split"] = split_name
    row["selection_score"] = float(0.5 * row["v_measure"] + 0.5 * row["pairwise_f1"])
    return row


def candidate_rows(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    split: Split,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray]:
    rows = []
    labels_by_key: dict[str, np.ndarray] = {}
    for k in [30, 50, 75]:
        for resolution in [1.0, 1.5, 2.0]:
            result = leiden_cluster(embeddings, k=k, resolution=resolution, seed=seed)
            labels = result.labels
            key = f"k={k};resolution={resolution};seed={seed}"
            labels_by_key[key] = labels
            rows.append(
                evaluate_subset(
                    df,
                    labels,
                    split.val_idx,
                    "validation_selection",
                    {
                        "experiment": "semi_supervised_embedding",
                        "backend": "torch_linear_projection_contrastive",
                        "method": "leiden",
                        "params": key,
                        "k": k,
                        "resolution": resolution,
                        "seed": seed,
                        "modularity": result.modularity,
                        "selected": False,
                    },
                )
            )
    candidates = (
        pd.DataFrame(rows)
        .sort_values("selection_score", ascending=False)
        .reset_index(drop=True)
    )
    best = candidates.iloc[0].to_dict()
    best["selected"] = True
    selected_labels = labels_by_key[str(best["params"])]
    return candidates, best, selected_labels


def write_note(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    manifest: dict[str, object],
    output: str | Path,
) -> None:
    holdout = metrics[
        (metrics["split"].eq("holdout")) & (metrics["selected"].eq(True))
    ].iloc[0]
    full = metrics[
        (metrics["split"].eq("full_corpus_label_guided"))
        & (metrics["selected"].eq(True))
    ].iloc[0]
    lines = [
        "# Semi-supervised embedding upper-bound",
        "",
        "This experiment is intentionally separate from the main unsupervised Leiden",
        "result. Macro-tags are used on the train and validation splits to shape the",
        "representation and select clustering parameters. Therefore the resulting",
        "metrics are not independent external validation.",
        "",
        "The local run trained a lightweight contrastive projection head over the saved",
        "BGE/PCA embeddings. It created positive pairs from jokes sharing at least one",
        "macro-tag, random negative pairs from jokes with disjoint macro-tags, and hard",
        "negative pairs from nearest neighbors with disjoint macro-tags.",
        "",
        "## Pair sampling",
        "",
        f"- Positive pairs: {manifest['pair_stats']['positive_pairs']}",
        f"- Negative pairs: {manifest['pair_stats']['negative_pairs']}",
        f"- Hard negative pairs: {manifest['pair_stats']['hard_negative_pairs']}",
        f"- Total pairs: {manifest['pair_stats']['total_pairs']}",
        "",
        "## Selected validation configuration",
        "",
        f"- Params: `{manifest['selected_params']}`",
        f"- Validation selection score: {manifest['validation_selection_score']:.4f}",
        "",
        "## Holdout and full-corpus metrics",
        "",
        f"- Holdout ARI: {holdout['ari']:.4f}",
        f"- Holdout V-measure: {holdout['v_measure']:.4f}",
        f"- Holdout pairwise F1: {holdout['pairwise_f1']:.4f}",
        f"- Full-corpus label-guided ARI: {full['ari']:.4f}",
        f"- Full-corpus label-guided V-measure: {full['v_measure']:.4f}",
        f"- Full-corpus label-guided pairwise F1: {full['pairwise_f1']:.4f}",
        "",
        "## Methodological warning",
        "",
        "These values should be described as an upper-bound or label-guided control.",
        "They must not replace the main unsupervised clustering metrics because the",
        "tag labels influenced the learned representation and validation selection.",
    ]
    if not manifest["sentence_transformers_available"]:
        lines.extend(
            [
                "",
                "The current local environment did not have `sentence-transformers`",
                "installed at run time, so the committed artifact uses the lightweight",
                "projection-head fallback rather than a full SentenceTransformer encoder",
                "fine-tune. The distinction is recorded in the manifest.",
            ]
        )
    lines.extend(["", "See also `outputs/tables/semi_supervised_vs_unsupervised.csv`."])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clustered", default="data/processed/anekdots_tagged_clustered.csv"
    )
    parser.add_argument("--pca", default="data/embeddings/tagged_pca128.npy")
    parser.add_argument(
        "--embedding-output", default="data/embeddings/tagged_bge_m3_finetuned.npy"
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/tables/semi_supervised_embedding_metrics.csv",
    )
    parser.add_argument(
        "--comparison-output",
        default="outputs/tables/semi_supervised_vs_unsupervised.csv",
    )
    parser.add_argument(
        "--note", default="outputs/report_notes/15_semi_supervised_upper_bound.md"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.clustered)
    features = normalize(np.load(args.pca), norm="l2").astype("float32", copy=False)
    tag_sets = macro_sets(df["macro_tags"])
    split = split_indices(tag_sets, args.seed)
    pairs, pair_stats = sample_pairs(
        features,
        tag_sets,
        split.train_idx,
        seed=args.seed,
        positives_per_anchor=2,
        random_negatives_per_anchor=2,
        hard_negatives_per_anchor=1,
    )
    embeddings, training_stats = train_projection(
        features,
        pairs,
        seed=args.seed,
        output_dim=args.output_dim,
        epochs=args.epochs,
        batch_size=512,
        learning_rate=1e-3,
        margin=0.2,
    )
    Path(args.embedding_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embedding_output, embeddings)

    candidates, best, selected_labels = candidate_rows(df, embeddings, split, args.seed)
    selected_rows = [
        evaluate_subset(
            df,
            selected_labels,
            split.holdout_idx,
            "holdout",
            {
                "experiment": "semi_supervised_embedding",
                "backend": "torch_linear_projection_contrastive",
                "method": "leiden",
                "params": best["params"],
                "k": best["k"],
                "resolution": best["resolution"],
                "seed": args.seed,
                "modularity": best["modularity"],
                "selected": True,
            },
        ),
        evaluate_subset(
            df,
            selected_labels,
            np.arange(len(df)),
            "full_corpus_label_guided",
            {
                "experiment": "semi_supervised_embedding",
                "backend": "torch_linear_projection_contrastive",
                "method": "leiden",
                "params": best["params"],
                "k": best["k"],
                "resolution": best["resolution"],
                "seed": args.seed,
                "modularity": best["modularity"],
                "selected": True,
            },
        ),
    ]
    metrics = pd.concat([candidates, pd.DataFrame(selected_rows)], ignore_index=True)
    metrics.loc[
        metrics["params"].eq(best["params"])
        & metrics["split"].eq("validation_selection"),
        "selected",
    ] = True
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        args.metrics_output, index=False, encoding="utf-8", lineterminator="\n"
    )

    unsup_labels = df["cluster_final"].to_numpy()
    comparison_rows = [
        evaluate_subset(
            df,
            unsup_labels,
            split.holdout_idx,
            "holdout",
            {
                "experiment": "unsupervised_leiden_final",
                "backend": "main_text_only_hybrid_features",
                "method": "leiden",
                "params": "k=75;resolution=2.0;seed=7",
                "k": 75,
                "resolution": 2.0,
                "seed": 7,
                "modularity": np.nan,
                "selected": True,
            },
        ),
        evaluate_subset(
            df,
            unsup_labels,
            np.arange(len(df)),
            "full_corpus_independent_validation",
            {
                "experiment": "unsupervised_leiden_final",
                "backend": "main_text_only_hybrid_features",
                "method": "leiden",
                "params": "k=75;resolution=2.0;seed=7",
                "k": 75,
                "resolution": 2.0,
                "seed": 7,
                "modularity": np.nan,
                "selected": True,
            },
        ),
        *selected_rows,
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        args.comparison_output, index=False, encoding="utf-8", lineterminator="\n"
    )

    split_df = pd.DataFrame(
        {
            "id": df["id"],
            "semi_supervised_split": "unassigned",
        }
    )
    split_df.loc[split.train_idx, "semi_supervised_split"] = "train"
    split_df.loc[split.val_idx, "semi_supervised_split"] = "validation"
    split_df.loc[split.holdout_idx, "semi_supervised_split"] = "holdout"
    split_df.to_csv(
        "outputs/tables/semi_supervised_split_assignments.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    manifest = {
        "embedding_file": args.embedding_output,
        "source_embedding_file": args.pca,
        "backend": "torch_linear_projection_contrastive",
        "label_guided": True,
        "independent_external_validation": False,
        "sentence_transformers_available": importlib.util.find_spec(
            "sentence_transformers"
        )
        is not None,
        "train_rows": int(len(split.train_idx)),
        "validation_rows": int(len(split.val_idx)),
        "holdout_rows": int(len(split.holdout_idx)),
        "pair_stats": pair_stats,
        "training_stats": training_stats,
        "selected_params": str(best["params"]),
        "validation_selection_score": float(best["selection_score"]),
        "note": (
            "This artifact is a label-guided upper-bound representation. The local run "
            "trained a contrastive projection over saved text embeddings, not the main "
            "unsupervised Leiden feature space."
        ),
    }
    manifest_path = Path(args.embedding_output).with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_note(metrics, comparison, manifest, args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
