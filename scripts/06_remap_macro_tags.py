from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from thesis_pipeline.tag_mapping import json_list, load_macro_mapping, map_tags, normalize_tag, parse_json_list


def append_progress(message: str) -> None:
    path = Path("outputs/report_notes/metric_improvement_progress_log.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().isoformat(timespec='seconds')}: {message}\n")


def count_macros(df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for values in df["macro_tags"].map(parse_json_list):
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    total_rows = len(df)
    return (
        pd.DataFrame([{"macro_tag": k, "count": v, "share": v / total_rows} for k, v in counts.items()])
        .sort_values(["count", "macro_tag"], ascending=[False, True])
        .reset_index(drop=True)
    )


def remap_df(df: pd.DataFrame, mapping_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = load_macro_mapping(mapping_path)
    rows = []
    unmapped = []
    for _, row in df.iterrows():
        tags = parse_json_list(row["tags_raw"])
        macros, missing = map_tags(tags, mapping)
        for tag in missing:
            unmapped.append({"tag_raw": tag, "tag_norm": normalize_tag(tag), "id": row["id"]})
        row = row.copy()
        row["tags_norm"] = json_list([normalize_tag(t) for t in tags])
        row["macro_tags"] = json_list(macros)
        row["macro_tag_count"] = len(macros)
        row["primary_macro_tag"] = macros[0]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(unmapped)


def write_mapping_csv(mapping_path: str) -> None:
    mapping = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8")) or {}
    records = []
    for macro, tags in mapping.items():
        for tag in tags or []:
            records.append({"macro_tag": macro, "tag_raw": tag, "tag_norm": normalize_tag(tag)})
    pd.DataFrame(records).to_csv("data/processed/tag_macro_mapping.csv", index=False, encoding="utf-8")


def write_audit(before: pd.DataFrame) -> None:
    dataset = pd.read_csv("data/processed/anekdots_tagged.csv")
    clustered = pd.read_csv("data/processed/anekdots_tagged_clustered.csv")
    single = pd.read_csv("outputs/tables/external_metrics_single_label.csv")
    pairwise = pd.read_csv("outputs/tables/pairwise_multilabel_metrics.csv")
    unmapped = pd.read_csv("outputs/tables/unmapped_tags.csv")
    nb = json.loads(Path("notebooks/tagged_corpus_analysis.ipynb").read_text(encoding="utf-8"))
    executed = sum(c.get("execution_count") is not None for c in nb.get("cells", []) if c.get("cell_type") == "code")
    outputs = sum(len(c.get("outputs", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
    required = [
        "outputs/tables/metrics_all.csv",
        "outputs/tables/feature_ablation_metrics.csv",
        "outputs/tables/clustering_search_all_runs.csv",
        "notebooks/tagged_corpus_analysis_execution_summary.ipynb",
    ]
    missing = [p for p in required if not Path(p).exists()]
    lines = [
        "# Audit before strong metric improvement",
        "",
        f"Dataset rows: **{len(dataset)}**.",
        f"Clustered rows: **{len(clustered)}**.",
        f"Current cluster count: **{clustered['cluster_leiden'].nunique()}**.",
        "",
        "## Current external metrics",
        single.to_markdown(index=False),
        "",
        "## Current sampled pairwise metrics",
        pairwise.to_markdown(index=False),
        "",
        "## Current top macro-tags",
        before.to_markdown(index=False),
        "",
        "## Current top unmapped tags",
        unmapped.head(40).to_markdown(index=False),
        "",
        f"Notebook `notebooks/tagged_corpus_analysis.ipynb`: executed code cells={executed}, output cells={outputs}.",
        "",
        "## Missing artifacts for this goal",
        "\n".join(f"- `{p}`" for p in missing) if missing else "None.",
        "",
        "## Planned commands",
        "```bash",
        "python scripts/06_remap_macro_tags.py",
        "python scripts/07_compute_exact_metrics.py",
        "python scripts/08_feature_ablation_and_search.py",
        "python scripts/09_select_final_and_interpret.py",
        "pytest -q",
        "python scripts/run_all.py --skip-existing",
        "```",
    ]
    Path("outputs/report_notes/05_audit_before_strong_metric_improvement.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    Path("outputs/report_notes").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    before = pd.read_csv("outputs/tables/top_macro_tags.csv")
    if "share" not in before.columns:
        before["share"] = before["count"] / pd.read_csv("data/processed/anekdots_tagged.csv").shape[0]
    write_audit(before)
    mapping_path = "config/tag_macro_categories.yml"
    for path in [
        "data/processed/anekdots_tagged.csv",
        "data/raw/anekdot_tagged_candidates.csv",
        "data/processed/anekdots_tagged_clustered.csv",
    ]:
        if Path(path).exists():
            df = pd.read_csv(path)
            remapped, unmapped = remap_df(df, mapping_path)
            remapped.to_csv(path, index=False, encoding="utf-8")
    write_mapping_csv(mapping_path)
    tagged = pd.read_csv("data/processed/anekdots_tagged.csv")
    after = count_macros(tagged)
    after.to_csv("outputs/tables/top_macro_tags.csv", index=False, encoding="utf-8")
    after.to_csv("outputs/tables/tag_distribution_macro.csv", index=False, encoding="utf-8")
    _, unmapped_after = remap_df(tagged, mapping_path)
    if unmapped_after.empty:
        unmapped_summary = pd.DataFrame(columns=["tag_raw", "count"])
    else:
        unmapped_summary = (
            unmapped_after.groupby("tag_raw", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        )
    unmapped_summary.to_csv("outputs/tables/unmapped_tags.csv", index=False, encoding="utf-8")
    unmapped_summary.to_csv("outputs/tables/unmapped_tags_after_remap.csv", index=False, encoding="utf-8")
    before_cmp = before.rename(columns={"count": "count_before", "share": "share_before"})
    after_cmp = after.rename(columns={"count": "count_after", "share": "share_after"})
    compare = before_cmp.merge(after_cmp, on="macro_tag", how="outer").fillna(0)
    compare["count_delta"] = compare["count_after"] - compare["count_before"]
    compare["share_delta"] = compare["share_after"] - compare["share_before"]
    compare.to_csv("outputs/tables/top_macro_tags_before_after.csv", index=False, encoding="utf-8")
    other_before = float(before.loc[before["macro_tag"] == "other", "count"].sum())
    other_after = float(after.loc[after["macro_tag"] == "other", "count"].sum())
    reduction = (other_before - other_after) / other_before if other_before else 0.0
    lines = [
        "# Strong macro-tag remap",
        "",
        "The remap is semantic and taxonomy-level: tags were assigned by meaning, not by cluster membership.",
        "No tag fields were used as embedding or clustering features.",
        "",
        f"`other` before: **{int(other_before)}** occurrences.",
        f"`other` after: **{int(other_after)}** occurrences.",
        f"Relative reduction: **{reduction:.1%}**.",
        "",
        "## Top macro-tags after remap",
        after.head(40).to_markdown(index=False),
        "",
        "## Remaining unmapped tags",
        unmapped_summary.head(40).to_markdown(index=False) if not unmapped_summary.empty else "No unmapped tags remain in the current corpus.",
    ]
    Path("outputs/report_notes/06_macro_tag_remap_strong.md").write_text("\n".join(lines), encoding="utf-8")
    append_progress(f"Phase 2 remap complete: other {int(other_before)} -> {int(other_after)} ({reduction:.1%} reduction).")
    print({"other_before": int(other_before), "other_after": int(other_after), "reduction": reduction, "unmapped_after": len(unmapped_summary)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
