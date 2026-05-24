from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def code_cell(source: str):
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "\n".join(
                [
                    "# Tagged corpus analysis execution summary",
                    "",
                    "This notebook is a local reproducibility summary based on saved repository artifacts.",
                    "It is not labeled as a Colab execution artifact because it is executed with nbclient in the local environment.",
                ]
            )
        ),
        code_cell(
            """
import importlib.util
import platform
import subprocess
import sys

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
if importlib.util.find_spec("torch"):
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
else:
    print("Torch: not installed")
    print("CUDA available: False")
    print("GPU: none")
try:
    print("Git commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
except Exception as exc:
    print("Git commit: unavailable", type(exc).__name__, exc)
            """
        ),
        code_cell(
            """
from pathlib import Path
import numpy as np
import pandas as pd

dataset_path = Path("data/processed/anekdots_tagged.csv")
clustered_path = Path("data/processed/anekdots_tagged_clustered.csv")
embedding_path = Path("data/embeddings/tagged_bge_m3.npy")
pca_path = Path("data/embeddings/tagged_pca128.npy")
final_metrics_path = Path("outputs/tables/final_metrics_summary.csv")
before_after_path = Path("outputs/tables/metrics_before_after.csv")
selection_path = Path("outputs/tables/final_clustering_selection.csv")

dataset = pd.read_csv(dataset_path)
clustered = pd.read_csv(clustered_path)
embeddings = np.load(embedding_path)
pca = np.load(pca_path)
final_metrics = pd.read_csv(final_metrics_path)
before_after = pd.read_csv(before_after_path)
selection = pd.read_csv(selection_path)

print("Read:", dataset_path, len(dataset))
print("Read:", clustered_path, len(clustered))
print("Embeddings shape:", embeddings.shape)
print("PCA shape:", pca.shape)
print("Final metrics rows:", len(final_metrics))
print("Before/after rows:", len(before_after))
print("Selection rows:", len(selection))
            """
        ),
        code_cell(
            """
selected = selection.iloc[0]
cluster_count = int(clustered["cluster_final"].nunique())
largest_cluster_share = float(clustered["cluster_final"].value_counts(normalize=True).max())

print("Final method:", selected["method"])
print("Final feature set:", selected["feature_set"])
print("Final params:", selected["params"])
print("Final cluster count:", cluster_count)
print("Largest cluster share:", round(largest_cluster_share, 6))
            """
        ),
        code_cell(
            """
final_all = final_metrics[(final_metrics["model"] == "final") & (final_metrics["subset"] == "all")].iloc[0]
single = final_metrics[(final_metrics["model"] == "final") & (final_metrics["subset"] == "single_clear_label")].iloc[0]

print("Final ARI:", round(float(final_all["ari"]), 6))
print("Final AMI:", round(float(final_all["ami"]), 6))
print("Final V-measure:", round(float(final_all["v_measure"]), 6))
print("Final pairwise precision:", round(float(final_all["pairwise_precision"]), 6))
print("Final pairwise recall:", round(float(final_all["pairwise_recall"]), 6))
print("Final pairwise F1:", round(float(final_all["pairwise_f1"]), 6))
print("Single-clear-label V-measure:", round(float(single["v_measure"]), 6))
print("Single-clear-label pairwise F1:", round(float(single["pairwise_f1"]), 6))
print(before_after[["metric", "old_leiden", "final", "delta"]].to_string(index=False))
            """
        ),
        code_cell(
            """
generated_report_artifacts = [
    "outputs/tables/final_clustering_selection.csv",
    "outputs/tables/final_metrics_summary.csv",
    "outputs/tables/final_internal_cluster_metrics.csv",
    "outputs/tables/metrics_before_after.csv",
    "outputs/tables/cluster_ctfidf_terms.csv",
    "outputs/tables/cluster_interpretation_cards.csv",
    "outputs/tables/cluster_report_ready_summary.csv",
    "outputs/tables/macro_tag_mapping_audit.csv",
    "outputs/report_notes/10_final_clustering_selection.md",
    "outputs/report_notes/11_cluster_interpretation.md",
    "outputs/report_notes/12_macro_tag_mapping_audit.md",
    "outputs/figures/umap2d_final.html",
    "outputs/figures/umap3d_final.html",
    "outputs/figures/umap2d_final.png",
    "outputs/figures/umap3d_final.png",
]
for artifact in generated_report_artifacts:
    path = Path(artifact)
    print(artifact, "exists=", path.exists(), "bytes=", path.stat().st_size if path.exists() else 0)
            """
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata["execution_note"] = "Executed locally with nbclient; no Colab/GPU values are fabricated."
    return nb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="notebooks/tagged_corpus_analysis_execution_summary.ipynb")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": "."}})
    client.execute()
    _, nb = nbformat.validator.normalize(nb)
    nbformat.write(nb, output)
    print({"notebook": str(output), "cells": len(nb.cells)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
