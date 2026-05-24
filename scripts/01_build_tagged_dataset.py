from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis_pipeline.anekdot_parser import BuildSettings, build_tagged_dataset
from thesis_pipeline.tag_mapping import parse_json_list
from thesis_pipeline.text_normalization import truncate_text


def tag_counts(df: pd.DataFrame, column: str, output_name: str) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for values in df[column].map(parse_json_list):
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    rows = [{"tag": key, "count": value} for key, value in counts.items()]
    return (
        pd.DataFrame(rows)
        .sort_values(["count", "tag"], ascending=[False, True])
        .rename(columns={"tag": output_name})
    )


def write_report(
    df: pd.DataFrame,
    stats: dict[str, object],
    tables: dict[str, pd.DataFrame],
    top_tags: pd.DataFrame,
    top_macro: pd.DataFrame,
    path: Path,
    command: str,
) -> None:
    zero = tables["coverage"].loc[
        tables["coverage"]["candidate_count_tagged_month"] == 0, ["year", "month"]
    ]
    examples = df.head(10).copy()
    examples["text_preview"] = examples["text"].map(lambda x: truncate_text(x, 300))
    lines = [
        "# Dataset rebuild: tagged-only anekdot.ru corpus",
        "",
        "## Parser settings",
        "```json",
        json.dumps(stats["settings"], ensure_ascii=False, indent=2),
        "```",
        "",
        f"Final row count: **{stats['row_count']}**.",
        f"Rows before exact normalized deduplication: **{stats['row_count_before_dedup']}**.",
        f"Exact duplicate rows collapsed: **{stats['duplicate_count']}**.",
        f"Months covered: **{stats['months_with_selected']} / {stats['months_total']}**.",
        f"Months with zero tagged jokes in parsed monthly archive: **{stats['months_zero_tagged']}**.",
        "",
        "The parser uses monthly anekdot.ru archive pages and keeps only joke containers with at least one",
        "site tag in the same `topicbox`. If a month has more than the configured maximum, rows are sampled",
        "deterministically with the configured seed; otherwise all tagged monthly candidates are kept.",
        "",
        "## Months with zero tagged jokes",
        zero.to_markdown(index=False) if not zero.empty else "No zero-tag months.",
        "",
        "## Per-year coverage",
        tables["year_coverage"].to_markdown(index=False),
        "",
        "## Top 30 raw tags",
        top_tags.head(30).to_markdown(index=False),
        "",
        "## Top macro categories",
        top_macro.to_markdown(index=False),
        "",
        "## Example rows",
        examples[
            ["id", "year", "month", "tags_raw", "macro_tags", "text_preview"]
        ].to_markdown(index=False),
        "",
        "## Limitation",
        "This is a tagged subcorpus. It is biased toward topics that anekdot.ru tags consistently and is",
        "not directly comparable with the old untagged 8,941-row corpus without explicitly discussing the",
        "sampling change.",
        "",
        "## Reproduce",
        "```bash",
        command,
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-month", type=int, default=3)
    parser.add_argument("--max-per-month", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--output", default="data/processed/anekdots_tagged.csv")
    parser.add_argument("--tag-map", default="config/tag_macro_categories.yml")
    args = parser.parse_args()

    for folder in [
        "data/raw",
        "data/processed",
        "outputs/tables",
        "outputs/report_notes",
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    settings = BuildSettings(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        max_per_month=args.max_per_month,
        seed=args.seed,
        sleep=args.sleep,
    )
    df, stats, tables = build_tagged_dataset(settings, args.tag_map)
    if df.empty:
        raise SystemExit("No tagged jokes were parsed.")
    if not (df["tag_count"] >= 1).all():
        raise SystemExit("Parser produced rows without tags.")
    if df["id"].duplicated().any():
        raise SystemExit("Parser produced duplicate ids.")

    output = Path(args.output)
    df.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    df.to_csv(
        "data/raw/anekdot_tagged_candidates.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    tables["tag_dictionary"].to_csv(
        "data/processed/tag_dictionary.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    tables["coverage"].to_csv(
        "outputs/tables/dataset_month_coverage.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    tables["year_coverage"].to_csv(
        "outputs/tables/dataset_year_coverage.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    tables["duplicates"].to_csv(
        "outputs/tables/duplicate_groups.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    top_tags = tag_counts(df, "tags_raw", "tag_raw")
    top_macro = tag_counts(df, "macro_tags", "macro_tag")
    top_tags.to_csv(
        "outputs/tables/top_tags.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    top_macro.to_csv(
        "outputs/tables/top_macro_tags.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    mapping_rows = []
    import yaml

    mapping = yaml.safe_load(Path(args.tag_map).read_text(encoding="utf-8"))
    for macro, tags in mapping.items():
        for tag in tags:
            mapping_rows.append({"macro_tag": macro, "tag_raw": tag})
    pd.DataFrame(mapping_rows).to_csv(
        "data/processed/tag_macro_mapping.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    command = (
        "python scripts/01_build_tagged_dataset.py "
        f"--start-year {args.start_year} --start-month {args.start_month} "
        f"--end-year {args.end_year} --end-month {args.end_month} "
        f"--max-per-month {args.max_per_month} --seed {args.seed} --sleep {args.sleep} "
        f"--output {args.output}"
    )
    write_report(
        df,
        stats,
        tables,
        top_tags,
        top_macro,
        Path("outputs/report_notes/00_dataset_rebuild.md"),
        command,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
