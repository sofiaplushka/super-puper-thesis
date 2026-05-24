from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis_pipeline.diagnostics import (
    dataset_bias,
    interpretability_summary,
    length_tables,
    near_duplicates,
    robustness_summary,
    save_year_month_figures,
    structural_features,
    structural_summaries,
)


def write_report(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Слабые места практической части и улучшения",
        "",
        "## Самые слабые места практической части",
        "- P0: Новый корпус является tagged-only и не должен описываться как прямое продолжение старого 8,941-row корпуса.",
        "- P0: 3D UMAP является визуализацией, а не доказательством качества кластеризации.",
        "- P1: Теги anekdot.ru являются silver labels, а не экспертной разметкой.",
        "- P1: Многоярлычная природа шуток снижает интерпретируемость single-label метрик.",
        "",
        "## Что уже исправлено кодом",
        "- Все downstream-шаги используют `data/processed/anekdots_tagged.csv`.",
        "- Кластеризация выполняется в embedding/PCA-пространстве.",
        "- UMAP 2D/3D сохранены только как визуализации.",
        "- Добавлены tag-based, internal и stability metrics.",
        "- Добавлены центральные и пограничные примеры кластеров.",
        "",
        "## Что стоит честно описать как ограничение",
        tables["bias"].to_markdown(index=False),
        "",
        "## Что можно добавить в отчёт без новых вычислений",
        "Использовать таблицы coverage, cluster interpretability, validation metrics и weakness diagnostics.",
        "",
        "## Что требует дополнительного эксперимента",
        "- Сравнение с экспертной ручной разметкой малой контрольной выборки.",
        "- Проверка других embedding-моделей.",
        "- Отдельная оценка влияния near-duplicates.",
        "",
        "## Готовые формулировки для ВКР",
        "В практической части был построен новый корпус, содержащий только шутки с тегами anekdot.ru.",
        "Такой выбор позволил использовать теги как внешние тематические silver labels для оценки кластеров.",
        "Ограничение подхода состоит в том, что тегированный подкорпус может быть смещён в сторону тем,",
        "которые на сайте размечались чаще и последовательнее.",
        "Качество кластеризации оценивалось не только визуально, но и через согласованность с макро-тегами,",
        "pairwise multilabel metrics, internal metrics и stability analysis.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="data/processed/anekdots_tagged_clustered.csv"
    )
    parser.add_argument("--validation-dir", default="outputs/tables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for folder in ["outputs/tables", "outputs/figures", "outputs/report_notes"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    save_year_month_figures(df)
    bias = dataset_bias(df)
    duplicates = near_duplicates(df)
    length_cluster, length_macro = length_tables(df)
    features = structural_features(df)
    structural_cluster, structural_macro = structural_summaries(features)
    purity_path = Path(args.validation_dir) / "cluster_tag_purity_entropy.csv"
    central_path = Path(args.validation_dir) / "cluster_central_examples.csv"
    borderline_path = Path(args.validation_dir) / "cluster_borderline_examples.csv"
    purity = pd.read_csv(purity_path) if purity_path.exists() else pd.DataFrame()
    central = pd.read_csv(central_path) if central_path.exists() else pd.DataFrame()
    borderline = (
        pd.read_csv(borderline_path) if borderline_path.exists() else pd.DataFrame()
    )
    interpretability = interpretability_summary(df, purity, central, borderline)
    robustness = robustness_summary(args.validation_dir)

    bias.to_csv(
        "outputs/tables/weakness_dataset_bias.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    duplicates.to_csv(
        "outputs/tables/weakness_duplicate_examples.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    length_cluster.to_csv(
        "outputs/tables/weakness_length_by_cluster.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    length_macro.to_csv(
        "outputs/tables/weakness_length_by_macro_tag.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    interpretability.to_csv(
        "outputs/tables/cluster_interpretability_summary.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    structural_cluster.to_csv(
        "outputs/tables/structural_features_by_cluster.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    structural_macro.to_csv(
        "outputs/tables/structural_features_by_macro_tag.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    robustness.to_csv(
        "outputs/tables/weakness_robustness_summary.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    write_report(
        Path("outputs/report_notes/04_weaknesses_and_improvements.md"), {"bias": bias}
    )
    print({"rows": len(df), "weakness_tables": 8})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
