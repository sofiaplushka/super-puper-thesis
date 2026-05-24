from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from thesis_pipeline.text_normalization import text_hash


def checksum_hashes(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_dataset(path: str | Path, text_column: str, id_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")
    if id_column not in df.columns:
        raise ValueError(f"Missing id column: {id_column}")
    if "text_norm_hash" not in df.columns:
        df["text_norm_hash"] = df[text_column].map(text_hash)
    return df


def compute_embeddings(
    texts: list[str],
    model_id: str,
    batch_size: int,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    from sentence_transformers import SentenceTransformer

    started = time.monotonic()
    model = SentenceTransformer(model_id)
    device = str(model.device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32", copy=False)
    elapsed = time.monotonic() - started
    tokenizer = getattr(model, "tokenizer", None)
    max_seq_length = getattr(model, "max_seq_length", None)
    token_lengths = []
    truncated = 0
    if tokenizer is not None:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            token_lengths.append(len(ids))
            if max_seq_length and len(ids) > int(max_seq_length):
                truncated += 1
    meta = {
        "device": device,
        "runtime_seconds": round(elapsed, 3),
        "max_seq_length": max_seq_length,
        "token_lengths": token_lengths,
        "truncated_text_count": truncated,
    }
    return embeddings, meta


def compute_pca(embeddings: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    if embeddings.shape[1] <= n_components:
        return embeddings.astype("float32", copy=False), {
            "skipped": True,
            "reason": f"embedding_dim <= {n_components}",
            "explained_variance_ratio_sum": None,
        }
    pca = PCA(n_components=n_components, random_state=seed)
    values = pca.fit_transform(embeddings).astype("float32", copy=False)
    return values, {
        "skipped": False,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "components": n_components,
    }


def write_manifest(path: str | Path, data: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_matches(path: str | Path, expected: dict[str, object]) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        current = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    keys = [
        "dataset_path",
        "dataset_row_count",
        "model_id",
        "normalized",
        "seed",
        "text_hash_checksum",
    ]
    return all(current.get(k) == expected.get(k) for k in keys)


def write_embedding_report(
    path: str | Path,
    manifest: dict[str, object],
    text_stats: dict[str, object],
    command: str,
) -> None:
    lines = [
        "# Embeddings",
        "",
        f"Dataset: `{manifest['dataset_path']}`",
        f"Rows: **{manifest['dataset_row_count']}**",
        f"Model: `{manifest['model_id']}`",
        f"Device: `{manifest.get('device')}`",
        f"Embedding shape: `{manifest['embedding_shape']}`",
        f"PCA shape: `{manifest['pca_shape']}`",
        f"Normalized: `{manifest['normalized']}`",
        f"Runtime seconds: `{manifest.get('runtime_seconds')}`",
        f"PCA explained variance: `{manifest.get('pca_explained_variance_ratio_sum')}`",
        f"Tokenizer max sequence length: `{manifest.get('max_seq_length')}`",
        f"Texts truncated by tokenizer: `{manifest.get('truncated_text_count')}`",
        "",
        "## Text length statistics",
        "```json",
        json.dumps(text_stats, ensure_ascii=False, indent=2),
        "```",
        "",
        "No character-level truncation is applied by the pipeline. If truncation is needed,",
        "it is handled by the model tokenizer and counted above.",
        "",
        "## Reproduce",
        "```bash",
        command,
        "```",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def length_stats(series: pd.Series) -> dict[str, object]:
    chars = series.fillna("").astype(str).map(len)
    words = series.fillna("").astype(str).map(lambda x: len(x.split()))
    return {
        "chars": {
            "min": int(chars.min()),
            "median": float(chars.median()),
            "mean": float(chars.mean()),
            "max": int(chars.max()),
        },
        "words": {
            "min": int(words.min()),
            "median": float(words.median()),
            "mean": float(words.mean()),
            "max": int(words.max()),
        },
    }


def save_text_hashes(df: pd.DataFrame, id_column: str, output_path: str | Path) -> None:
    df[[id_column, "text_norm_hash"]].rename(columns={id_column: "id"}).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def expected_manifest(
    dataset_path: str,
    df: pd.DataFrame,
    model_id: str,
    normalized: bool,
    seed: int,
) -> dict[str, object]:
    hashes = df["text_norm_hash"].astype(str).tolist()
    return {
        "dataset_path": dataset_path,
        "dataset_row_count": int(len(df)),
        "model_id": model_id,
        "normalized": normalized,
        "seed": seed,
        "text_hash_checksum": checksum_hashes(hashes),
    }


def complete_manifest(
    expected: dict[str, object],
    embeddings: np.ndarray,
    pca: np.ndarray,
    model_meta: dict[str, object],
    pca_meta: dict[str, object],
) -> dict[str, object]:
    token_lengths = model_meta.get("token_lengths") or []
    token_stats = None
    if token_lengths:
        arr = np.asarray(token_lengths)
        token_stats = {
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "max": int(arr.max()),
        }
    manifest = dict(expected)
    manifest.update(
        {
            "embedding_shape": list(embeddings.shape),
            "pca_shape": list(pca.shape),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": model_meta.get("device"),
            "runtime_seconds": model_meta.get("runtime_seconds"),
            "max_seq_length": model_meta.get("max_seq_length"),
            "token_length_stats": token_stats,
            "truncated_text_count": model_meta.get("truncated_text_count"),
            "pca_explained_variance_ratio_sum": pca_meta.get("explained_variance_ratio_sum"),
        }
    )
    return manifest

