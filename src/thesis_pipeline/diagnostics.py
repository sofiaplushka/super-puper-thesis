from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from rapidfuzz import fuzz

from thesis_pipeline.tag_mapping import parse_json_list
from thesis_pipeline.text_normalization import normalize_text, truncate_text


def save_year_month_figures(df: pd.DataFrame) -> None:
    year = df.groupby("year", as_index=False).size().rename(columns={"size": "rows"})
    month = df.groupby(["year", "month"], as_index=False).size().rename(columns={"size": "rows"})
    px.bar(year, x="year", y="rows", title="Tagged corpus coverage by year").write_html(
        "outputs/figures/year_coverage.html", include_plotlyjs="cdn", full_html=True
    )
    month["period"] = month["year"].astype(str) + "-" + month["month"].astype(str).str.zfill(2)
    px.line(month, x="period", y="rows", title="Tagged corpus coverage by month").write_html(
        "outputs/figures/month_coverage.html", include_plotlyjs="cdn", full_html=True
    )


def dataset_bias(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"issue": "row_count", "value": len(df), "severity": "info"})
    rows.append({"issue": "year_min", "value": int(df["year"].min()), "severity": "info"})
    rows.append({"issue": "year_max", "value": int(df["year"].max()), "severity": "info"})
    rows.append({"issue": "largest_year_share", "value": float(df["year"].value_counts(normalize=True).max()), "severity": "P1"})
    macro_counts = {}
    for values in df["macro_tags"].map(parse_json_list):
        for value in values:
            macro_counts[value] = macro_counts.get(value, 0) + 1
    if macro_counts:
        rows.append({"issue": "largest_macro_share", "value": max(macro_counts.values()) / len(df), "severity": "P1"})
    return pd.DataFrame(rows)


def near_duplicates(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    exact = df[df["text_norm_hash"].duplicated(keep=False)].copy()
    records = []
    for _, row in exact.head(limit).iterrows():
        records.append({"kind": "exact", "id": row["id"], "similar_id": None, "score": 100, "text_preview": truncate_text(row["text"], 220)})
    if len(records) >= limit:
        return pd.DataFrame(records)
    values = df[["id", "text"]].copy()
    values["norm"] = values["text"].map(normalize_text)
    values["block"] = values["norm"].str[:24].str.casefold()
    for _, group in values.groupby("block"):
        if len(group) < 2 or len(records) >= limit:
            continue
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                score = fuzz.ratio(rows[i]["norm"], rows[j]["norm"])
                if score >= 92:
                    records.append(
                        {
                            "kind": "near",
                            "id": rows[i]["id"],
                            "similar_id": rows[j]["id"],
                            "score": score,
                            "text_preview": truncate_text(rows[i]["text"], 220),
                        }
                    )
                if len(records) >= limit:
                    break
            if len(records) >= limit:
                break
    return pd.DataFrame(records)


def length_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_cluster = df.groupby("cluster_leiden").agg(
        rows=("id", "count"),
        chars_mean=("text_length_chars", "mean"),
        chars_median=("text_length_chars", "median"),
        words_mean=("text_length_words", "mean"),
        words_max=("text_length_words", "max"),
    ).reset_index()
    rows = []
    for _, row in df.iterrows():
        for macro in parse_json_list(row["macro_tags"]):
            rows.append({"macro_tag": macro, "text_length_chars": row["text_length_chars"], "text_length_words": row["text_length_words"]})
    macro_df = pd.DataFrame(rows)
    by_macro = macro_df.groupby("macro_tag").agg(
        rows=("macro_tag", "count"),
        chars_mean=("text_length_chars", "mean"),
        chars_median=("text_length_chars", "median"),
        words_mean=("text_length_words", "mean"),
        words_max=("text_length_words", "max"),
    ).reset_index()
    px.histogram(df, x="text_length_chars", nbins=60, title="Text length distribution").write_html(
        "outputs/figures/text_length_distribution.html", include_plotlyjs="cdn", full_html=True
    )
    return by_cluster, by_macro


def structural_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["id", "cluster_leiden", "macro_tags", "text"]].copy()
    result["line_count"] = result["text"].fillna("").astype(str).map(lambda x: max(1, x.count("\n") + 1))
    result["sentence_count"] = result["text"].fillna("").astype(str).map(lambda x: max(1, len(re.findall(r"[.!?]+", x))))
    result["dash_count"] = result["text"].fillna("").astype(str).map(lambda x: x.count("-") + x.count("—"))
    result["question_count"] = result["text"].fillna("").astype(str).map(lambda x: x.count("?"))
    result["exclamation_count"] = result["text"].fillna("").astype(str).map(lambda x: x.count("!"))
    result["looks_like_dialogue"] = result["dash_count"] >= 2
    return result


def structural_summaries(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["line_count", "sentence_count", "dash_count", "question_count", "exclamation_count", "looks_like_dialogue"]
    by_cluster = features.groupby("cluster_leiden")[numeric].mean().reset_index()
    rows = []
    for _, row in features.iterrows():
        for macro in parse_json_list(row["macro_tags"]):
            item = {k: row[k] for k in numeric}
            item["macro_tag"] = macro
            rows.append(item)
    by_macro = pd.DataFrame(rows).groupby("macro_tag")[numeric].mean().reset_index()
    return by_cluster, by_macro


def interpretability_summary(df: pd.DataFrame, purity: pd.DataFrame, central: pd.DataFrame, borderline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, part in df.groupby("cluster_leiden"):
        pur = purity[purity["cluster_leiden"] == cluster]
        dominant = pur["dominant_macro_tag"].iloc[0] if not pur.empty and "dominant_macro_tag" in pur else None
        share = float(pur["dominant_macro_share"].iloc[0]) if not pur.empty and "dominant_macro_share" in pur else 0.0
        confidence = "high" if share >= 0.65 else "medium" if share >= 0.4 else "low"
        rows.append(
            {
                "cluster_leiden": cluster,
                "size": len(part),
                "dominant_macro_tag": dominant,
                "dominant_macro_share": share,
                "suggested_cluster_name": f"{dominant or 'mixed'} cluster",
                "interpretation_confidence": confidence,
                "central_example_ids": ", ".join(central.loc[central["cluster_leiden"] == cluster, "id"].astype(str).head(5)),
                "borderline_example_ids": ", ".join(borderline.loc[borderline["cluster_leiden"] == cluster, "id"].astype(str).head(5)),
                "caveat": "Mixed tags and humor formulas may blur this cluster." if confidence != "high" else "Dominant macro tag is relatively concentrated.",
            }
        )
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def robustness_summary(tables_dir: str | Path) -> pd.DataFrame:
    grid_path = Path(tables_dir) / "leiden_stability_grid.csv"
    pair_path = Path(tables_dir) / "leiden_stability_pairwise.csv"
    rows = []
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        rows.append({"issue": "cluster_count_min", "value": int(grid["cluster_count"].min()), "severity": "info"})
        rows.append({"issue": "cluster_count_max", "value": int(grid["cluster_count"].max()), "severity": "P1"})
        rows.append({"issue": "largest_cluster_share_max", "value": float(grid["largest_cluster_share"].max()), "severity": "P1"})
    if pair_path.exists():
        pair = pd.read_csv(pair_path)
        rows.append({"issue": "stability_ari_mean", "value": float(pair["ari"].mean()), "severity": "P1"})
        rows.append({"issue": "stability_ami_mean", "value": float(pair["ami"].mean()), "severity": "P1"})
    return pd.DataFrame(rows)

