from pathlib import Path

import nbformat


NOTEBOOK = Path("notebooks/tagged_corpus_analysis_execution_summary.ipynb")


def test_execution_summary_notebook_is_real_executed_artifact():
    assert NOTEBOOK.exists()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.get("outputs") for cell in code_cells)
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    assert any(count is not None for count in execution_counts)
    assert not all(count is None for count in execution_counts)


def test_execution_summary_reads_required_repository_files():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(str(cell.get("source", "")) for cell in nb.cells)
    required_paths = [
        "data/processed/anekdots_tagged.csv",
        "data/processed/anekdots_tagged_clustered.csv",
        "data/embeddings/tagged_bge_m3.npy",
        "data/embeddings/tagged_pca128.npy",
        "outputs/tables/final_metrics_summary.csv",
        "outputs/tables/metrics_before_after.csv",
        "outputs/tables/final_clustering_selection.csv",
    ]
    missing = [path for path in required_paths if path not in source]
    assert missing == []


def test_notebook_does_not_make_fake_colab_claim():
    assert not Path("notebooks/tagged_corpus_analysis_executed_colab.ipynb").exists()
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "executed_colab" not in text
    assert "local reproducibility summary" in text
