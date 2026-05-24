import math
from pathlib import Path

import pandas as pd


def test_final_internal_metrics_reflect_final_clustering():
    path = Path("outputs/tables/final_internal_cluster_metrics.csv")
    assert path.exists()
    metrics = pd.read_csv(path)
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["method"] == "leiden"
    assert row["feature_set"] == "hybrid_dense_lexical_dw0.75_lw0.25"
    assert int(row["cluster_count"]) == 20
    assert math.isclose(
        float(row["largest_cluster_share"]),
        0.12906153566890544,
        rel_tol=0,
        abs_tol=1e-9,
    )
    assert str(row["modularity"]).strip() not in {"", "nan", "None"}
    assert row["cluster_label_column"] == "cluster_final"


def test_legacy_internal_metrics_now_points_to_final_and_initial_is_archived():
    final_metrics = pd.read_csv("outputs/tables/final_internal_cluster_metrics.csv")
    current_metrics = pd.read_csv("outputs/tables/internal_cluster_metrics.csv")
    assert (
        int(current_metrics.iloc[0]["cluster_count"])
        == int(final_metrics.iloc[0]["cluster_count"])
        == 20
    )
    assert Path("outputs/tables/internal_cluster_metrics_initial_leiden.csv").exists()
