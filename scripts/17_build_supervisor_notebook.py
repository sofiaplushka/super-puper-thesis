from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = [
        md(
            """
            # Кластеризация анекдотов anekdot.ru

            Этот ноутбук собирает итоговую практическую часть в одном месте:
            загрузку корпуса, разведочный анализ, визуализацию кластеров,
            основные метрики, и дополнительные контрольные эксперименты.

            Главный результат остается без учителя: Leiden-кластеризация по
            текстовым признакам. Теги используются только для оценки и
            отдельных контрольных моделей, где это явно указано.
            """
        ),
        code(
            """
            from pathlib import Path
            import subprocess
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import IFrame, Markdown, display

            ROOT = Path.cwd()
            if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
                ROOT = ROOT.parent

            pd.set_option("display.max_columns", 80)
            pd.set_option("display.max_colwidth", 120)
            print(f"Рабочая папка: {ROOT}")
            """
        ),
        md(
            """
            ## Воспроизводимый запуск

            По умолчанию ноутбук читает уже сохраненные результаты, чтобы его
            можно было быстро показать и прогнать. Если нужно пересобрать
            артефакты, переключите `RUN_PIPELINE` на `True`. Тяжелый шаг с
            эмбеддингами можно пропустить, если файлы уже есть.
            """
        ),
        code(
            """
            RUN_PIPELINE = False

            commands = [
                [sys.executable, "scripts/run_all.py", "--skip-existing"],
                [sys.executable, "scripts/13_hierarchical_evaluation.py"],
                [sys.executable, "scripts/14_supervised_tag_prediction_baseline.py"],
                [sys.executable, "scripts/15_semi_supervised_upper_bound.py"],
                [sys.executable, "scripts/16_final_evaluation_story.py"],
            ]

            for cmd in commands:
                print(" ".join(cmd))
                if RUN_PIPELINE:
                    subprocess.run(cmd, cwd=ROOT, check=True)
            """
        ),
        md(
            """
            ## Данные

            Итоговый корпус содержит только записи, у которых на сайте есть
            хотя бы один тег. Это важно: такие теги дают внешний источник
            проверки качества без ручной построчной разметки анекдотов.
            """
        ),
        code(
            """
            dataset = pd.read_csv(ROOT / "data/processed/anekdots_tagged.csv")
            clustered = pd.read_csv(ROOT / "data/processed/anekdots_tagged_clustered.csv")
            print(f"Строк в корпусе: {len(dataset):,}".replace(",", " "))
            print(f"Период: {dataset['year'].min()}-{dataset['month'].min():02d} ... {dataset['year'].max()}-{dataset['month'].max():02d}")
            print(f"Месяцев с данными: {dataset[['year', 'month']].drop_duplicates().shape[0]}")
            print(f"Строк с тегами: {(dataset['tag_count'] >= 1).sum():,}".replace(",", " "))
            display(dataset[["id", "year", "month", "text_clean", "tags_raw", "macro_tags"]].head(5))
            """
        ),
        md(
            """
            ## Разведочный анализ корпуса

            Ниже показаны распределение по годам и самые частотные макро-теги.
            Макро-теги нужны для оценки результата, но не добавляются в текст и
            не используются как признаки основного кластерного алгоритма.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            dataset["year"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#4b8bbe")
            axes[0].set_title("Количество анекдотов по годам")
            axes[0].set_xlabel("Год")
            axes[0].set_ylabel("Строк")
            axes[0].tick_params(axis="x", labelrotation=90)

            top_macro = pd.read_csv(ROOT / "outputs/tables/top_macro_tags.csv").head(15)
            axes[1].barh(top_macro["macro_tag"][::-1], top_macro["count"][::-1], color="#609966")
            axes[1].set_title("Самые частотные макро-теги")
            axes[1].set_xlabel("Упоминаний")
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Основная кластеризация

            Главный результат: Leiden-кластеризация в гибридном текстовом
            пространстве BGE/PCA + TF-IDF/SVD. Параметры финальной модели:
            `k=75`, `resolution=2.0`, `seed=7`. Всего получилось 20 кластеров.
            """
        ),
        code(
            """
            final_metrics = pd.read_csv(ROOT / "outputs/tables/final_metrics_summary.csv")
            final_main = final_metrics[(final_metrics["model"] == "final") & (final_metrics["subset"] == "all")]
            display(final_main[[
                "method", "feature_set", "params", "rows", "cluster_count",
                "largest_cluster_share", "ari", "ami", "v_measure",
                "pairwise_precision", "pairwise_recall", "pairwise_f1"
            ]])

            sizes = pd.read_csv(ROOT / "outputs/tables/cluster_final_sizes.csv")
            ax = sizes.sort_values("size", ascending=False).plot(
                x="cluster_final", y="size", kind="bar", figsize=(12, 4), legend=False, color="#8064a2"
            )
            ax.set_title("Размеры финальных кластеров")
            ax.set_xlabel("Кластер")
            ax.set_ylabel("Строк")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Интерактивная карта UMAP

            UMAP используется только для просмотра структуры, а не для обучения
            кластеров. Если интерактивный блок не отображается в просмотрщике,
            откройте файл `outputs/figures/umap3d_final.html`.
            """
        ),
        code(
            """
            html_path = ROOT / "outputs/figures/umap3d_final.html"
            display(IFrame(src=str(html_path), width="100%", height=620))
            """
        ),
        md(
            """
            ## Иерархическая оценка

            Подробные макро-теги были дополнительно объединены в широкие
            тематические группы первого уровня. Такая оценка обычно выше,
            потому что она проверяет более грубое тематическое совпадение.
            """
        ),
        code(
            """
            hierarchical = pd.read_csv(ROOT / "outputs/tables/hierarchical_metrics_summary.csv")
            display(hierarchical[[
                "label_level", "subset", "rows", "label_count", "ari", "ami",
                "nmi", "homogeneity", "completeness", "v_measure", "pairwise_f1"
            ]])
            """
        ),
        md(
            """
            ## Контрольная модель с учителем

            Эта модель не является кластеризацией. Она предсказывает макро-теги
            по текстовым признакам на разбиении train/validation/test. Цель -
            проверить, есть ли в тексте сигнал, связанный с тегами.
            """
        ),
        code(
            """
            supervised = pd.read_csv(ROOT / "outputs/tables/supervised_tag_prediction_baseline.csv")
            supervised_test = supervised[supervised["split"] == "test"].sort_values("micro_f1", ascending=False)
            display(supervised_test[[
                "feature_set", "macro_f1", "micro_f1", "weighted_f1",
                "micro_precision", "micro_recall", "subset_accuracy"
            ]])
            """
        ),
        md(
            """
            ## Полуобучаемая верхняя граница

            В этом эксперименте теги уже влияют на представление текстов:
            создаются пары похожих и непохожих анекдотов, затем обучается
            контрастивное преобразование эмбеддингов. Поэтому эти метрики нельзя
            считать независимой проверкой основной кластеризации.
            """
        ),
        code(
            """
            semi = pd.read_csv(ROOT / "outputs/tables/semi_supervised_embedding_metrics.csv")
            display(semi[semi["selected"] == True][[
                "split", "params", "rows", "cluster_count", "ari", "ami",
                "v_measure", "pairwise_precision", "pairwise_recall", "pairwise_f1"
            ]])
            """
        ),
        md(
            """
            ## Итоговая картина метрик

            В дипломе основной строкой следует считать независимую
            unsupervised-кластеризацию. Остальные строки - вспомогательные:
            они объясняют, почему грубые темы восстанавливаются лучше и почему
            модели с использованием тегов дают отдельную, не независимую оценку.
            """
        ),
        code(
            """
            story = pd.read_csv(ROOT / "outputs/tables/final_evaluation_story.csv")
            display(story[[
                "display_name", "method_type", "evaluation_scope",
                "main_or_auxiliary", "independent_external_validation",
                "v_measure", "pairwise_f1", "macro_f1", "micro_f1", "note"
            ]])
            """
        ),
        md(
            """
            ## Краткий вывод

            Основная модель дает умеренные, но честные метрики: ARI около
            0.277, V-measure около 0.387, точная pairwise F1 около 0.339.
            Это ожидаемо для коротких юмористических текстов с несколькими
            темами и шумными пользовательскими тегами. Иерархическая оценка
            показывает согласование на более широких темах, а модели с
            использованием тегов подтверждают, что теговый сигнал в тексте есть,
            но они должны описываться отдельно от независимой кластеризации.
            """
        ),
    ]
    return nb


def execute_notebook(path: Path) -> None:
    try:
        from nbclient import NotebookClient
    except Exception as exc:
        print(f"Notebook was written but not executed: nbclient unavailable ({exc})")
        return
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    nbf.write(nb, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="notebooks/Sophie_анеки_кластеризация_итоговая.ipynb")
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output)
    if not args.no_execute:
        execute_notebook(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
