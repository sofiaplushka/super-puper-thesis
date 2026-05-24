import math

import pandas as pd

from thesis_pipeline.strong_metrics import exact_pairwise_multilabel, metric_suite


def test_exact_pairwise_multilabel_counts_overlap_pairs():
    labels = [0, 0, 1, 1]
    tag_sets = [{"x"}, {"x", "y"}, {"z"}, {"y"}]

    result = exact_pairwise_multilabel(labels, tag_sets)

    assert result.total_pairs == 6
    assert result.model_positive_pairs == 2
    assert result.label_positive_pairs == 2
    assert result.true_positive == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.true_negative == 3
    assert math.isclose(result.precision, 0.5)
    assert math.isclose(result.recall, 0.5)
    assert math.isclose(result.f1, 0.5)


def test_metric_suite_reports_expected_subsets():
    df = pd.DataFrame(
        {
            "macro_tags": [
                '["politics_power"]',
                '["politics_power", "military_conflict"]',
                '["other"]',
                '["family_relationships"]',
            ]
        }
    )
    metrics = metric_suite(df, [0, 0, 1, 2])

    assert set(metrics["subset"]) == {"all", "excluding_other", "single_clear_label"}
    assert int(metrics.loc[metrics["subset"].eq("all"), "rows"].iloc[0]) == 4
    assert int(metrics.loc[metrics["subset"].eq("excluding_other"), "rows"].iloc[0]) == 3
    assert int(metrics.loc[metrics["subset"].eq("single_clear_label"), "rows"].iloc[0]) == 2
