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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

