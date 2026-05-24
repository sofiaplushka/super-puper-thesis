from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis_pipeline.tag_mapping import (
    load_macro_mapping,
    macro_lookup,
    normalize_tag,
    parse_json_list,
)

FORMAL_MACROS = {"textual_forms", "internet_memes", "consumer_services"}
FORMAL_TAG_HINTS = {
    "британские ученые",
    "объявления",
    "инструкции",
    "цитаты",
    "реклама",
    "русский язык",
    "мобильный",
    "windows",
}
BROAD_MACROS = {
    "everyday_life",
    "absurd_philosophy",
    "politics_power",
    "work_professions",
}
BROAD_TAG_HINTS = {"о жизни", "работа", "политика", "деньги", "новые русские", "кризис"}
DEBATABLE_TAG_HINTS = {
    "армянское радио",
    "ржевский",
    "чапаев",
    "рабинович",
    "шовинизм",
    "блондинки",
    "брюнетки",
    "пошлые",
    "звёзды",
    "звезды",
}


def status_for(raw_tag: str, macro: str) -> tuple[str, str]:
    norm = normalize_tag(raw_tag)
    if norm in DEBATABLE_TAG_HINTS:
        return (
            "debatable",
            "A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation.",
        )
    if macro in FORMAL_MACROS or norm in FORMAL_TAG_HINTS:
        return (
            "formal_or_technical",
            "The tag denotes a text form, medium, service, or technical object rather than a topical gold label.",
        )
    if macro in BROAD_MACROS or norm in BROAD_TAG_HINTS:
        return (
            "broad_category",
            "The observed tag is broad, so the mapping is intentionally coarse and should be interpreted cautiously.",
        )
    if macro == "other":
        return (
            "debatable",
            "No responsible semantic macro-category was available; this tag remains residual.",
        )
    return (
        "confident",
        "The raw tag has a direct semantic match to the assigned macro-category.",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/anekdots_tagged.csv")
    parser.add_argument("--mapping", default="config/tag_macro_categories.yml")
    parser.add_argument(
        "--output", default="outputs/tables/macro_tag_mapping_audit.csv"
    )
    parser.add_argument(
        "--note", default="outputs/report_notes/12_macro_tag_mapping_audit.md"
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.note).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    mapping = load_macro_mapping(args.mapping)
    lookup = macro_lookup(mapping)
    raw_counts: Counter[str] = Counter()
    display_by_norm: dict[str, str] = {}
    examples: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        row_id = str(row["id"])
        for raw_tag in parse_json_list(row["tags_raw"]):
            norm = normalize_tag(raw_tag)
            if not norm:
                continue
            raw_counts[norm] += 1
            display_by_norm.setdefault(norm, raw_tag)
            if len(examples[norm]) < 5:
                examples[norm].append(row_id)

    rows = []
    for norm, count in sorted(raw_counts.items(), key=lambda item: (-item[1], item[0])):
        raw_tag = display_by_norm[norm]
        macro = lookup.get(norm, "other")
        audit_status, rationale = status_for(raw_tag, macro)
        rows.append(
            {
                "raw_tag": raw_tag,
                "raw_tag_count": int(count),
                "mapped_macro_tag": macro,
                "audit_status": audit_status,
                "audit_rationale": rationale,
                "example_joke_ids": ";".join(examples[norm]),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")

    status_counts = (
        audit["audit_status"]
        .value_counts()
        .rename_axis("audit_status")
        .reset_index(name="count")
    )
    macro_counts = (
        audit["mapped_macro_tag"]
        .value_counts()
        .rename_axis("mapped_macro_tag")
        .reset_index(name="raw_tag_count")
    )
    debatable = audit[audit["audit_status"].eq("debatable")].head(40)
    other_count = int(audit["mapped_macro_tag"].eq("other").sum())
    note_lines = [
        "# Macro-tag mapping audit",
        "",
        f"Observed raw tags audited: **{len(audit)}**.",
        f"Raw tags still mapped to `other`: **{other_count}**.",
        "",
        "The `other` bucket became zero in the processed corpus because the mapping is now exhaustive at the raw-tag level.",
        "This is not an expert gold-standard relabeling of individual jokes. It is a tag-level taxonomy: each observed site tag is assigned to a semantic macro-category before validation metrics are computed.",
        "",
        "No individual joke rows were manually relabeled. The audit records raw tag counts and example ids only to make the tag-level decisions reviewable.",
        "",
        "This affects ARI, AMI, and V-measure by replacing a large residual label with more specific silver-label categories.",
        "The metrics therefore measure alignment with the constructed macro-tag taxonomy, not agreement with expert topic labels.",
        "",
        "## Audit status counts",
        "",
        status_counts.to_markdown(index=False),
        "",
        "## Most common mapped macro-tags by raw-tag vocabulary",
        "",
        macro_counts.head(30).to_markdown(index=False),
        "",
        "## Debatable mappings to review",
        "",
        debatable[
            [
                "raw_tag",
                "raw_tag_count",
                "mapped_macro_tag",
                "audit_rationale",
                "example_joke_ids",
            ]
        ].to_markdown(index=False),
    ]
    Path(args.note).write_text("\n".join(note_lines), encoding="utf-8")
    print(
        json.dumps(
            {"raw_tags": len(audit), "other_raw_tags": other_count}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
