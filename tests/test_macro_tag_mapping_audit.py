from pathlib import Path

import pandas as pd

from thesis_pipeline.tag_mapping import normalize_tag, parse_json_list


def test_macro_tag_mapping_audit_covers_all_observed_raw_tags():
    df = pd.read_csv("data/processed/anekdots_tagged.csv", usecols=["id", "tags_raw"])
    observed = {
        normalize_tag(tag)
        for values in df["tags_raw"].map(parse_json_list)
        for tag in values
        if normalize_tag(tag)
    }
    audit = pd.read_csv("outputs/tables/macro_tag_mapping_audit.csv")
    audited = set(audit["raw_tag"].map(normalize_tag))
    assert observed == audited
    assert len(audit) == len(observed) == 158


def test_macro_tag_mapping_audit_statuses_and_other_bucket_are_explicit():
    audit = pd.read_csv("outputs/tables/macro_tag_mapping_audit.csv")
    allowed = {"confident", "debatable", "formal_or_technical", "broad_category"}
    assert set(audit["audit_status"]).issubset(allowed)
    assert not audit["audit_rationale"].isna().any()
    assert not audit["example_joke_ids"].isna().any()
    assert int(audit["mapped_macro_tag"].eq("other").sum()) == 0
    note = Path("outputs/report_notes/12_macro_tag_mapping_audit.md").read_text(encoding="utf-8")
    assert "exhaustive at the raw-tag level" in note
    assert "No individual joke rows were manually relabeled" in note
