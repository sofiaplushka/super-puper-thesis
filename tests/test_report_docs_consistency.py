from pathlib import Path


def test_report_update_guide_separates_baseline_and_final_results():
    text = Path("docs/report_update_guide.md").read_text(encoding="utf-8")
    required_sections = [
        "## Historical baseline, not final",
        "## Final dataset",
        "## Final clustering configuration",
        "## Final metrics to report",
        "## What not to overstate",
        "## Russian thesis-ready wording",
    ]
    missing = [section for section in required_sections if section not in text]
    assert missing == []
    assert "initial Leiden run produced 12 clusters" in text
    assert "Do not insert" in text
    assert "Final clusters: 20" in text
    assert "0.2768" in text and "0.3871" in text and "0.3388" in text
    assert "supervised classification quality" in text


def test_execution_report_and_readme_reference_final_pipeline():
    execution_report = Path("docs/execution_report.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "notebooks/tagged_corpus_analysis_execution_summary.ipynb" in execution_report
    assert "tagged_corpus_analysis_executed_colab.ipynb" not in execution_report
    assert "final_internal_cluster_metrics.csv" in execution_report
    for script_name in [
        "scripts/06_remap_macro_tags.py",
        "scripts/07_compute_exact_metrics.py",
        "scripts/08_feature_ablation_and_search.py",
        "scripts/09_select_final_and_interpret.py",
    ]:
        assert script_name in readme
