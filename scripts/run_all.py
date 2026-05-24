from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], skip: bool = False) -> None:
    if skip:
        print("SKIP", " ".join(cmd))
        return
    print("RUN", " ".join(cmd))
    subprocess.check_call(cmd)


def exists_all(paths: list[str]) -> bool:
    return all(Path(p).exists() for p in paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    run(
        [py, "scripts/01_build_tagged_dataset.py"],
        skip=args.skip_existing and Path("data/processed/anekdots_tagged.csv").exists(),
    )
    run(
        [py, "scripts/02_compute_embeddings.py", "--mode", "local", "--skip-existing"],
        skip=args.skip_embeddings or (args.skip_existing and exists_all(["data/embeddings/tagged_bge_m3.npy", "data/embeddings/tagged_pca128.npy"])),
    )
    run(
        [py, "scripts/03_cluster_and_visualize.py"],
        skip=args.skip_existing and Path("data/processed/anekdots_tagged_clustered.csv").exists(),
    )
    run([py, "scripts/04_validate_clusters_with_tags.py"])
    run([py, "scripts/05_analyze_practical_weaknesses.py"])
    run(
        [py, "scripts/06_remap_macro_tags.py"],
        skip=args.skip_existing
        and exists_all(
            [
                "outputs/tables/top_macro_tags_before_after.csv",
                "outputs/tables/unmapped_tags_after_remap.csv",
                "outputs/report_notes/06_macro_tag_remap_strong.md",
            ]
        ),
    )
    run(
        [py, "scripts/07_compute_exact_metrics.py"],
        skip=args.skip_existing
        and exists_all(
            [
                "outputs/tables/metrics_all.csv",
                "outputs/tables/pairwise_multilabel_metrics_exact.csv",
                "outputs/report_notes/07_validation_metrics_exact.md",
            ]
        ),
    )
    run(
        [py, "scripts/08_feature_ablation_and_search.py"],
        skip=args.skip_existing
        and exists_all(
            [
                "outputs/tables/feature_ablation_metrics.csv",
                "outputs/tables/clustering_search_all_runs.csv",
                "outputs/report_notes/09_strong_clustering_search.md",
            ]
        ),
    )
    run(
        [py, "scripts/09_select_final_and_interpret.py"],
        skip=args.skip_existing
        and exists_all(
            [
                "outputs/tables/final_clustering_selection.csv",
                "outputs/tables/final_metrics_summary.csv",
                "outputs/report_notes/11_cluster_interpretation.md",
            ]
        ),
    )
    run(
        [py, "scripts/11_compute_final_internal_metrics.py"],
    )
    run(
        [py, "scripts/12_macro_tag_mapping_audit.py"],
        skip=args.skip_existing
        and exists_all(
            [
                "outputs/tables/macro_tag_mapping_audit.csv",
                "outputs/report_notes/12_macro_tag_mapping_audit.md",
            ]
        ),
    )
    run(
        [py, "scripts/10_build_execution_summary_notebook.py"],
        skip=args.skip_existing
        and Path("notebooks/tagged_corpus_analysis_execution_summary.ipynb").exists(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
