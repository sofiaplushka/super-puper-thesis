import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd


def test_run_all_supports_force_flag():
    result = subprocess.run(
        [sys.executable, "scripts/run_all.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--force" in result.stdout
    source = Path("scripts/run_all.py").read_text(encoding="utf-8")
    assert "skip_existing = args.skip_existing and not args.force" in source


def test_root_prompt_pack_ignores_are_present():
    text = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in [
        ".codex_goal/",
        "reference_prompts/",
        "PASTE_THIS_GOAL.md",
        "codex_metric_goal_pack/",
        "*_goal_prompts/",
        "*.prompt.md",
    ]:
        assert pattern in text


def test_supervisor_notebook_uses_real_year_month_period():
    df = pd.read_csv("data/processed/anekdots_tagged.csv", usecols=["year", "month"])
    periods = pd.PeriodIndex(
        year=df["year"].astype(int),
        month=df["month"].astype(int),
        freq="M",
    )
    expected = f"Период: {periods.min()} ... {periods.max()}"
    has_2026_12 = (
        (df["year"].astype(int) == 2026) & (df["month"].astype(int) == 12)
    ).any()

    nb = nbformat.read(
        "notebooks/Sophie_анеки_кластеризация_итоговая.ipynb", as_version=4
    )
    source = "\n".join(cell.source for cell in nb.cells)
    output_text = "\n".join(
        output.get("text", "")
        for cell in nb.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )

    assert "pd.PeriodIndex" in source
    assert expected in output_text
    if not has_2026_12:
        assert "Период: 1996-01 ... 2026-12" not in output_text


def test_supervisor_notebook_default_commands_are_fast_and_nonduplicated():
    source = Path("scripts/17_build_supervisor_notebook.py").read_text(encoding="utf-8")
    assert "RUN_PIPELINE = False" in source
    assert "RUN_AUXILIARY_CONTROLS = False" in source
    assert (
        source.count('[sys.executable, "scripts/run_all.py", "--skip-existing"]') == 1
    )
    assert "if RUN_AUXILIARY_CONTROLS:" in source


def test_csv_outputs_are_pandas_readable_and_lf_normalized():
    csv_paths = sorted(Path("outputs/tables").glob("*.csv"))
    assert csv_paths
    for path in csv_paths:
        pd.read_csv(path)
        data = path.read_bytes()
        assert b"\r\n" not in data, path
