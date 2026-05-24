from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from thesis_pipeline.embeddings import (
    complete_manifest,
    compute_embeddings,
    compute_pca,
    expected_manifest,
    length_stats,
    load_dataset,
    manifest_matches,
    save_text_hashes,
    write_embedding_report,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/anekdots_tagged.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=["local", "remote"], default="local")
    parser.add_argument("--output-dir", default="data/embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.mode == "remote":
        print("Remote mode is configured outside this script via the Colab bridge. Falling back to local execution in the current runtime.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path("outputs/report_notes").mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input, args.text_column, args.id_column)
    expected = expected_manifest(args.input, df, args.model, True, args.seed)
    manifest_path = out / "tagged_embeddings_manifest.json"
    emb_path = out / "tagged_bge_m3.npy"
    pca_path = out / "tagged_pca128.npy"
    ids_path = out / "tagged_ids.npy"
    hashes_path = out / "tagged_text_hashes.csv"
    if args.skip_existing and emb_path.exists() and pca_path.exists() and manifest_matches(manifest_path, expected):
        print(f"Reusing embeddings from {emb_path}")
        return 0

    texts = df[args.text_column].fillna("").astype(str).tolist()
    embeddings, model_meta = compute_embeddings(texts, args.model, args.batch_size, normalize=True)
    pca, pca_meta = compute_pca(embeddings, 128, args.seed)
    if embeddings.shape[0] != len(df) or pca.shape[0] != len(df):
        raise SystemExit("Embedding row count does not match dataset row count.")

    np.save(emb_path, embeddings)
    np.save(pca_path, pca)
    np.save(ids_path, df[args.id_column].astype(str).to_numpy())
    save_text_hashes(df, args.id_column, hashes_path)
    manifest = complete_manifest(expected, embeddings, pca, model_meta, pca_meta)
    write_manifest(manifest_path, manifest)
    command = (
        "python scripts/02_compute_embeddings.py "
        f"--input {args.input} --text-column {args.text_column} --id-column {args.id_column} "
        f"--model {args.model} --batch-size {args.batch_size} --mode {args.mode} "
        f"--output-dir {args.output_dir} --seed {args.seed}"
    )
    write_embedding_report(
        "outputs/report_notes/01_embeddings.md",
        manifest,
        length_stats(df[args.text_column]),
        command,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

