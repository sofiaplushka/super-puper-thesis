from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from thesis_pipeline.tag_mapping import json_list, load_macro_mapping, map_tags, normalize_tag
from thesis_pipeline.text_normalization import count_words, normalize_text, text_hash, truncate_text

BASE_URL = "https://www.anekdot.ru"
USER_AGENT = "Mozilla/5.0 (compatible; tagged-thesis-pipeline/1.0)"

ARCHIVE_COLUMNS = [
    "id",
    "joke_id",
    "url",
    "archive_url",
    "source_page",
    "year",
    "month",
    "date",
    "section",
    "genre",
    "text",
    "text_clean",
    "text_norm_hash",
    "text_length_chars",
    "text_length_words",
    "rating",
    "author",
    "author_url",
    "tags_raw",
    "tags_norm",
    "tag_count",
    "macro_tags",
    "macro_tag_count",
    "primary_macro_tag",
    "selected_rank_in_month",
    "candidate_count_tagged_month",
    "parsed_at",
]


@dataclass(frozen=True)
class BuildSettings:
    start_year: int = 1996
    start_month: int = 1
    end_year: int = 2026
    end_month: int = 3
    max_per_month: int = 30
    seed: int = 42
    sleep: float = 0.2


def iter_months(settings: BuildSettings) -> Iterable[tuple[int, int]]:
    year, month = settings.start_year, settings.start_month
    while (year, month) <= (settings.end_year, settings.end_month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def archive_urls(year: int, month: int) -> list[str]:
    yy = str(year)[-2:]
    mm = f"{month:02d}"
    return [
        f"{BASE_URL}/an/an{yy}{mm}/j{yy}{mm};100.html",
        f"{BASE_URL}/an/an{yy}{mm}/j{yy}{mm}.html",
    ]


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str | None:
    response = session.get(url, timeout=timeout)
    if response.status_code != 200:
        return None
    return response.text


def parse_archive_date(raw: str, fallback_year: int, fallback_month: int) -> str:
    match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", raw or "")
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"
    return f"{fallback_year}-{fallback_month:02d}-01"


def parse_rating(box) -> int | None:
    num = box.select_one(".votingbox .num")
    if num:
        try:
            return int(num.get_text(strip=True).replace("+", ""))
        except ValueError:
            return None
    rates = box.select_one(".rates")
    if rates and rates.get("data-r"):
        pieces = [p for p in rates["data-r"].split(";") if p]
        if pieces:
            try:
                return int(pieces[0])
            except ValueError:
                return None
    return None


def parse_box(box, archive_url: str, year: int, month: int) -> dict[str, object] | None:
    text_el = box.select_one("div.text")
    if not text_el:
        return None
    if box.get("data-t") and box.get("data-t") != "j":
        return None
    raw_tags = []
    for link in box.select("div.tags a[href*='/tags/']"):
        tag = normalize_text(link.get_text(" ", strip=True))
        href = (link.get("href") or "").rstrip("/")
        if tag and href != "/tags":
            raw_tags.append(tag)
    if not raw_tags:
        return None

    text = normalize_text(text_el.get_text("\n", strip=True))
    if not text:
        return None

    title_text = normalize_text((box.select_one("p.title") or box).get_text(" ", strip=True))
    joke_id = box.get("data-id") or ""
    permalink = f"{BASE_URL}/id/{joke_id}/" if joke_id else archive_url
    author = box.select_one("a.auth")
    raw_tags = list(dict.fromkeys(raw_tags))
    norm_tags = [normalize_tag(t) for t in raw_tags]
    stable_id = joke_id if joke_id else text_hash(f"{year}-{month}-{text}")[:16]
    return {
        "id": stable_id,
        "joke_id": joke_id or None,
        "url": permalink,
        "archive_url": archive_url,
        "source_page": title_text or archive_url,
        "year": year,
        "month": month,
        "date": parse_archive_date(title_text, year, month),
        "section": "archive",
        "genre": "anekdot",
        "text": text,
        "text_clean": text,
        "text_norm_hash": text_hash(text),
        "text_length_chars": len(text),
        "text_length_words": count_words(text),
        "rating": parse_rating(box),
        "author": normalize_text(author.get_text(" ", strip=True)) if author else None,
        "author_url": urljoin(BASE_URL, author.get("href")) if author and author.get("href") else None,
        "tags_raw_list": raw_tags,
        "tags_norm_list": norm_tags,
    }


def parse_archive_month(
    session: requests.Session,
    year: int,
    month: int,
) -> tuple[list[dict[str, object]], str | None, int]:
    last_url = None
    for url in archive_urls(year, month):
        last_url = url
        html = fetch_html(session, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        boxes = [b for b in soup.select("div.topicbox") if b.select_one("div.text")]
        for box in boxes:
            parsed = parse_box(box, url, year, month)
            if parsed:
                rows.append(parsed)
        return rows, url, len(boxes)
    return [], last_url, 0


def build_tag_dictionary(session: requests.Session) -> pd.DataFrame:
    html = fetch_html(session, f"{BASE_URL}/tags/")
    rows = []
    if html:
        soup = BeautifulSoup(html, "html.parser")
        seen = set()
        for link in soup.select("a[href*='/tags/']"):
            href = link.get("href") or ""
            tag = normalize_text(link.get_text(" ", strip=True))
            if not tag or href.rstrip("/") == "/tags":
                continue
            norm = normalize_tag(tag)
            if norm in seen:
                continue
            seen.add(norm)
            rows.append({"tag_raw": tag, "tag_norm": norm, "url": urljoin(BASE_URL, href)})
    return pd.DataFrame(rows).sort_values("tag_norm").reset_index(drop=True)


def choose_month_rows(rows: list[dict[str, object]], settings: BuildSettings) -> list[dict[str, object]]:
    if len(rows) <= settings.max_per_month:
        selected = list(rows)
    else:
        rng = random.Random(f"{settings.seed}:{rows[0]['year']}:{rows[0]['month']}")
        selected = rng.sample(rows, settings.max_per_month)
        selected.sort(key=lambda r: str(r["id"]))
    for idx, row in enumerate(selected, start=1):
        row["selected_rank_in_month"] = idx
        row["candidate_count_tagged_month"] = len(rows)
    return selected


def deduplicate_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], pd.DataFrame]:
    by_hash: dict[str, dict[str, object]] = {}
    duplicate_records = []
    for row in rows:
        key = str(row["text_norm_hash"])
        if key not in by_hash:
            by_hash[key] = row
            continue
        kept = by_hash[key]
        duplicate_records.append(
            {
                "text_norm_hash": key,
                "kept_id": kept["id"],
                "duplicate_id": row["id"],
                "kept_year": kept["year"],
                "duplicate_year": row["year"],
                "text_preview": truncate_text(str(row["text"]), 200),
            }
        )
        tags = list(dict.fromkeys(list(kept["tags_raw_list"]) + list(row["tags_raw_list"])))
        kept["tags_raw_list"] = tags
        kept["tags_norm_list"] = [normalize_tag(t) for t in tags]
    return list(by_hash.values()), pd.DataFrame(duplicate_records)


def finalize_rows(
    rows: list[dict[str, object]],
    mapping_path: str | Path,
    parsed_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = load_macro_mapping(mapping_path)
    parsed_at = parsed_at or datetime.now(timezone.utc).isoformat()
    unmapped_records = []
    final_rows = []
    for row in rows:
        tags_raw = list(row.pop("tags_raw_list"))
        tags_norm = list(row.pop("tags_norm_list"))
        macro_tags, unmapped = map_tags(tags_raw, mapping)
        for tag in unmapped:
            unmapped_records.append({"tag_raw": tag, "tag_norm": normalize_tag(tag), "id": row["id"]})
        row.update(
            {
                "tags_raw": json_list(tags_raw),
                "tags_norm": json_list(tags_norm),
                "tag_count": len(tags_raw),
                "macro_tags": json_list(macro_tags),
                "macro_tag_count": len(macro_tags),
                "primary_macro_tag": macro_tags[0],
                "parsed_at": parsed_at,
            }
        )
        final_rows.append(row)
    df = pd.DataFrame(final_rows)
    for column in ARCHIVE_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[ARCHIVE_COLUMNS].sort_values(["year", "month", "selected_rank_in_month", "id"]).reset_index(drop=True)
    unmapped_df = pd.DataFrame(unmapped_records)
    return df, unmapped_df


def build_tagged_dataset(
    settings: BuildSettings,
    mapping_path: str | Path,
    sleep: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    sleep_seconds = settings.sleep if sleep is None else sleep
    all_rows: list[dict[str, object]] = []
    coverage_rows = []
    for year, month in iter_months(settings):
        rows, archive_url, candidate_count_total = parse_archive_month(session, year, month)
        selected = choose_month_rows(rows, settings) if rows else []
        all_rows.extend(selected)
        coverage_rows.append(
            {
                "year": year,
                "month": month,
                "archive_url": archive_url,
                "candidate_count_total_month": candidate_count_total,
                "candidate_count_tagged_month": len(rows),
                "selected_count": len(selected),
            }
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    deduped, duplicates = deduplicate_rows(all_rows)
    df, unmapped = finalize_rows(deduped, mapping_path)
    tag_dictionary = build_tag_dictionary(session)
    coverage = pd.DataFrame(coverage_rows)
    year_coverage = coverage.groupby("year", as_index=False).agg(
        months=("month", "count"),
        selected_count=("selected_count", "sum"),
        tagged_candidates=("candidate_count_tagged_month", "sum"),
        total_candidates=("candidate_count_total_month", "sum"),
    )
    stats = {
        "settings": settings.__dict__,
        "row_count_before_dedup": len(all_rows),
        "row_count": len(df),
        "duplicate_count": len(duplicates),
        "months_total": len(coverage),
        "months_with_selected": int((coverage["selected_count"] > 0).sum()),
        "months_zero_tagged": int((coverage["candidate_count_tagged_month"] == 0).sum()),
    }
    tables = {
        "coverage": coverage,
        "year_coverage": year_coverage,
        "duplicates": duplicates,
        "tag_dictionary": tag_dictionary,
        "unmapped": unmapped,
    }
    return df, stats, tables

