from __future__ import annotations

import argparse
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
        md("""
            # Кластеризация анекдотов anekdot.ru

            Этот ноутбук повторяет логику старого `Sophie_анеки_кластеризация.ipynb`,
            но на финальном корпусе и финальных артефактах дипломной работы.

            Внутри есть не только итоговые метрики, но и ход работы: загрузка
            данных, проверка эмбеддингов, анализ близости текстов, сравнение
            простых кластерных подходов, Leiden + UMAP, сохранение таблицы с
            кластерами и дополнительные контрольные оценки.

            Главный результат остается кластеризацией без учителя. Теги не
            добавляются в текст и не используются как признаки основной модели.
            """),
        code("""
            from pathlib import Path
            import json
            import subprocess
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import HTML, IFrame, Image, Markdown, display
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import normalize

            ROOT = Path.cwd()
            if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
                ROOT = ROOT.parent

            rng = np.random.default_rng(42)
            pd.set_option("display.max_columns", 100)
            pd.set_option("display.max_colwidth", 140)
            plt.rcParams["figure.dpi"] = 120
            print(f"Рабочая папка: {ROOT}")
            """),
        md("""
            ## Воспроизводимый запуск

            По умолчанию ноутбук читает уже сохраненные артефакты, чтобы его
            можно было быстро открыть и показать научному руководителю. Если
            нужно пересобрать все легкие этапы, включите `RUN_PIPELINE = True`.

            Шаг с BGE-M3 эмбеддингами тяжелый и обычно запускается отдельно на
            GPU. В этом ноутбуке он не пересчитывается автоматически.
            """),
        code("""
            RUN_PIPELINE = False
            RUN_AUXILIARY_CONTROLS = False

            commands = [
                [sys.executable, "scripts/run_all.py", "--skip-existing"],
            ]
            auxiliary_commands = [
                [sys.executable, "scripts/13_hierarchical_evaluation.py"],
                [sys.executable, "scripts/14_supervised_tag_prediction_baseline.py"],
                [sys.executable, "scripts/15_semi_supervised_upper_bound.py"],
                [sys.executable, "scripts/16_final_evaluation_story.py"],
            ]

            if RUN_AUXILIARY_CONTROLS:
                commands.extend(auxiliary_commands)

            for cmd in commands:
                print(" ".join(cmd))
                if RUN_PIPELINE:
                    subprocess.run(cmd, cwd=ROOT, check=True)
            """),
        md("""
            ## Загрузка корпуса

            Финальная выборка отличается от старого `anekdots.csv`: здесь
            оставлены только анекдоты, у которых на сайте есть хотя бы один тег.
            Это нужно для внешней проверки кластеров без ручной построчной
            разметки.
            """),
        code("""
            dataset = pd.read_csv(ROOT / "data/processed/anekdots_tagged.csv")
            clustered = pd.read_csv(ROOT / "data/processed/anekdots_tagged_clustered.csv")
            periods = pd.PeriodIndex(
                year=dataset["year"].astype(int),
                month=dataset["month"].astype(int),
                freq="M",
            )

            print(f"Строк в корпусе: {len(dataset):,}".replace(",", " "))
            print(f"Строк с тегами: {(dataset['tag_count'] >= 1).sum():,}".replace(",", " "))
            print(f"Период: {periods.min()} ... {periods.max()}")
            print(f"Месяцев с данными: {dataset[['year', 'month']].drop_duplicates().shape[0]}")
            print(f"Финальных кластеров: {clustered['cluster_final'].nunique()}")

            display(dataset[["id", "year", "month", "text_clean", "tags_raw", "macro_tags"]].head(5))
            """),
        md("""
            ## Небольшой разведочный анализ

            Сначала посмотрим, насколько равномерно корпус покрывает годы и
            какие макро-теги встречаются чаще всего. Макро-теги дальше
            используются только для оценки и интерпретации, не для обучения
            основной кластеризации.
            """),
        code("""
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
            """),
        md("""
            ## Получение эмбеддингов

            В старом ноутбуке эмбеддинги считались прямо в ячейке. В финальной
            версии они уже сохранены как артефакты, чтобы не тратить время и
            GPU-кредиты при каждом показе. Используется модель `BAAI/bge-m3`.

            Исторически сравнивались несколько моделей:

            - `sentence-transformers/sentence-t5-base`
            - `intfloat/multilingual-e5-base`
            - `intfloat/multilingual-e5-large`
            - `BAAI/bge-m3`

            Для финального пайплайна оставлена `BAAI/bge-m3`, потому что она
            лучше подошла для русскоязычных коротких текстов и дала более
            пригодное пространство соседства.
            """),
        code("""
            embeddings = np.load(ROOT / "data/embeddings/tagged_bge_m3.npy")
            pca128 = np.load(ROOT / "data/embeddings/tagged_pca128.npy")
            ids = np.load(ROOT / "data/embeddings/tagged_ids.npy", allow_pickle=True)
            manifest = json.loads((ROOT / "data/embeddings/tagged_embeddings_manifest.json").read_text(encoding="utf-8"))

            print("Модель:", manifest["model_id"])
            print("Устройство при расчете:", manifest["device"])
            print("Время расчета, секунд:", manifest["runtime_seconds"])
            print("Форма эмбеддингов:", embeddings.shape)
            print("Форма PCA:", pca128.shape)
            print("Доля объясненной дисперсии PCA:", round(manifest["pca_explained_variance_ratio_sum"], 4))
            print("Обрезанных текстов токенизатором:", manifest["truncated_text_count"])
            print("ID синхронизированы с датасетом:", bool((ids.astype(str) == dataset["id"].astype(str).to_numpy()).all()))
            """),
        md("""
            ## Проверка синхронизации

            Это аналог `sanity_check` из старого ноутбука. Проверяем, что строки
            корпуса, эмбеддинги, PCA, UMAP и финальные метки кластеров
            согласованы между собой.
            """),
        code("""
            def sanity_check(df, clustered_df, embeddings_matrix, pca_matrix, ids_array):
                print("\\n=== SANITY CHECK ===")
                print("dataset rows:", len(df))
                print("clustered rows:", len(clustered_df))
                print("embeddings rows:", embeddings_matrix.shape[0])
                print("pca rows:", pca_matrix.shape[0])
                print("unique final clusters:", clustered_df["cluster_final"].nunique())
                print("largest cluster share:", round(clustered_df["cluster_final"].value_counts(normalize=True).max(), 4))
                print("missing texts:", int(df["text_clean"].isna().sum()))
                print("missing UMAP coordinates:", int(clustered_df[["umap2_x", "umap2_y", "umap3_x", "umap3_y", "umap3_z"]].isna().sum().sum()))

                assert len(df) == len(clustered_df) == embeddings_matrix.shape[0] == pca_matrix.shape[0] == len(ids_array)
                assert (ids_array.astype(str) == df["id"].astype(str).to_numpy()).all()
                assert clustered_df["cluster_final"].nunique() == 20
                assert clustered_df["feature_set"].nunique() == 1
                print("OK: данные согласованы")

            sanity_check(dataset, clustered, embeddings, pca128, ids)
            """),
        md("""
            ## Анализ пространства эмбеддингов

            Как и в старом ноутбуке, сравним косинусное сходство случайных пар
            и ближайших соседей. Если ближайшие соседи заметно ближе случайных
            пар, значит в эмбеддингах есть полезная структура соседства.
            """),
        code("""
            emb_norm = normalize(embeddings, norm="l2")
            n_pairs = 20_000
            a = rng.integers(0, len(emb_norm), size=n_pairs)
            b = rng.integers(0, len(emb_norm), size=n_pairs)
            keep = a != b
            a, b = a[keep], b[keep]
            random_sims = np.sum(emb_norm[a] * emb_norm[b], axis=1)

            nn = NearestNeighbors(n_neighbors=2, metric="cosine")
            nn.fit(emb_norm)
            nn_dist, nn_idx = nn.kneighbors(emb_norm)
            nearest_sims = 1 - nn_dist[:, 1]

            sim_summary = pd.DataFrame(
                {
                    "random_pairs": pd.Series(random_sims).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]),
                    "nearest_neighbors": pd.Series(nearest_sims).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]),
                }
            )
            display(sim_summary)
            print("Разница средних nearest - random:", round(float(nearest_sims.mean() - random_sims.mean()), 4))
            """),
        code("""
            plt.figure(figsize=(11, 5))
            plt.hist(random_sims, bins=100, density=True, alpha=0.45, label="Случайные пары", color="#7aa6c2")
            plt.hist(nearest_sims, bins=100, density=True, alpha=0.45, label="Ближайшие соседи", color="#d98c5f")
            plt.axvline(random_sims.mean(), color="#24577a", linestyle="--", linewidth=1)
            plt.axvline(nearest_sims.mean(), color="#a14f2b", linestyle="--", linewidth=1)
            plt.title("Косинусное сходство: случайные пары и ближайшие соседи")
            plt.xlabel("cosine similarity")
            plt.ylabel("Плотность")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """),
        md("""
            ## Почему не простой KMeans

            Старый ноутбук проверял KMeans и показывал, что для такого
            пространства он работает хуже. В финальном пайплайне это сохранено
            как сравнение моделей: KMeans и агломеративная кластеризация есть в
            таблице, но главным результатом выбран Leiden.
            """),
        code("""
            metrics = pd.read_csv(ROOT / "outputs/tables/final_metrics_summary.csv")
            comparison = (
                metrics[metrics["subset"].eq("all")]
                .loc[:, ["model", "method", "feature_set", "params", "cluster_count", "largest_cluster_share", "ari", "ami", "v_measure", "pairwise_f1"]]
                .sort_values(["model"])
            )
            display(comparison)
            """),
        md("""
            ## Теперь кластеризуем графом: Leiden + UMAP

            Кластеризация выполняется не в координатах UMAP. UMAP нужен только
            для визуального отображения уже найденных кластеров. Основная
            финальная конфигурация:

            - признаки: `hybrid_dense_lexical_dw0.75_lw0.25`
            - метод: Leiden
            - `k=75`, `resolution=2.0`, `seed=7`
            - число кластеров: 20
            """),
        code("""
            final_main = metrics[(metrics["model"] == "final") & (metrics["subset"] == "all")].iloc[0]
            display(pd.DataFrame([final_main])[[
                "method", "feature_set", "params", "rows", "cluster_count",
                "largest_cluster_share", "ari", "ami", "v_measure",
                "pairwise_precision", "pairwise_recall", "pairwise_f1"
            ]])
            """),
        code("""
            demo_export = ROOT / "outputs/tables/clustered_anekdots_for_supervisor.csv"
            export_columns = [
                "id", "year", "month", "text_clean", "tags_raw", "macro_tags",
                "cluster_final", "cluster_method", "feature_set",
                "umap2_x", "umap2_y", "umap3_x", "umap3_y", "umap3_z",
            ]
            clustered[export_columns].to_csv(demo_export, index=False, encoding="utf-8", lineterminator="\\n")
            print(f"Saved: {demo_export.relative_to(ROOT)}")
            """),
        code("""
            plt.figure(figsize=(12, 8))
            labels = clustered["cluster_final"].astype(str)
            for label in sorted(labels.unique(), key=lambda x: int(x)):
                part = clustered[labels == label]
                plt.scatter(part["umap2_x"], part["umap2_y"], s=8, alpha=0.65, label=label)

            plt.title("Leiden clustering + UMAP")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.grid(alpha=0.25)
            plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=8)
            plt.tight_layout()
            plt.show()
            """),
        md("""
            ## Интерактивная 3D-карта

            Эта карта удобна для демонстрации: можно вращать пространство и
            смотреть отдельные точки. Если в просмотрщике HTML не отобразится,
            откройте файл `outputs/figures/umap3d_final.html`.
            """),
        code("""
            html_path = ROOT / "outputs/figures/umap3d_final.html"
            png_path = ROOT / "outputs/figures/umap3d_final.png"

            print("HTML exists:", html_path.exists(), html_path)
            print("PNG exists:", png_path.exists(), png_path)

            if png_path.exists():
                display(Image(filename=str(png_path), width=900))

            if html_path.exists():
                html_url = html_path.resolve().as_uri()
                display(HTML(f'<p><a href="{html_url}" target="_blank">Открыть интерактивную 3D-карту в браузере</a></p>'))
                display(IFrame(src=html_url, width="100%", height=620))
            """),
        md("""
            ## Интерпретация кластеров

            Для отчета важны не только номера кластеров, но и их содержательное
            описание. Ниже показаны размеры кластеров и готовые карточки
            интерпретации.
            """),
        code("""
            sizes = pd.read_csv(ROOT / "outputs/tables/cluster_final_sizes.csv")
            cards = pd.read_csv(ROOT / "outputs/tables/cluster_final_interpretation_cards.csv")

            fig, ax = plt.subplots(figsize=(12, 4))
            sizes.sort_values("size", ascending=False).plot(
                x="cluster_final", y="size", kind="bar", legend=False, color="#8064a2", ax=ax
            )
            ax.set_title("Размеры финальных кластеров")
            ax.set_xlabel("Кластер")
            ax.set_ylabel("Строк")
            plt.tight_layout()
            plt.show()

            display(cards.head(12))
            """),
        md("""
            ## Иерархическая оценка

            Подробные макро-теги дополнительно объединены в широкие группы.
            Обычно такая оценка может быть выше, потому что она проверяет более
            грубое тематическое совпадение. В текущем прогоне она не стала выше
            для V-measure и pairwise F1: широкие группы оказались достаточно
            разнородными, поэтому это только вспомогательная проверка.
            """),
        code("""
            hierarchical = pd.read_csv(ROOT / "outputs/tables/hierarchical_metrics_summary.csv")
            display(hierarchical[[
                "label_level", "subset", "rows", "label_count", "ari", "ami",
                "nmi", "homogeneity", "completeness", "v_measure", "pairwise_f1"
            ]])
            """),
        md("""
            ## Контрольная модель с учителем

            Это не кластеризация. Здесь модель учится предсказывать макро-теги
            по текстовым признакам на разбиении train/validation/test. Такой
            baseline показывает, что теговый сигнал в тексте есть, но его нельзя
            выдавать за независимую оценку кластеризации.
            """),
        code("""
            supervised = pd.read_csv(ROOT / "outputs/tables/supervised_tag_prediction_baseline.csv")
            supervised_test = supervised[supervised["split"] == "test"].sort_values("micro_f1", ascending=False)
            display(supervised_test[[
                "feature_set", "macro_f1", "micro_f1", "weighted_f1",
                "micro_precision", "micro_recall", "subset_accuracy"
            ]])
            """),
        md("""
            ## Полуобучаемая верхняя граница

            В этом эксперименте теги уже влияют на представление текстов:
            создаются положительные и отрицательные пары, затем обучается
            контрастивное преобразование эмбеддингов. Поэтому это верхняя
            граница с подсказкой от тегов, а не независимая проверка.
            """),
        code("""
            semi = pd.read_csv(ROOT / "outputs/tables/semi_supervised_embedding_metrics.csv")
            display(semi[semi["selected"] == True][[
                "split", "params", "rows", "cluster_count", "ari", "ami",
                "v_measure", "pairwise_precision", "pairwise_recall", "pairwise_f1"
            ]])
            """),
        md("""
            ## Итоговая картина метрик

            Основной строкой для диплома остается `unsupervised_leiden_final`.
            Остальные строки нужны как поясняющие контрольные эксперименты.
            """),
        code("""
            story = pd.read_csv(ROOT / "outputs/tables/final_evaluation_story.csv")
            display(story[[
                "display_name", "method_type", "evaluation_scope",
                "main_or_auxiliary", "independent_external_validation",
                "v_measure", "pairwise_f1", "macro_f1", "micro_f1", "note"
            ]])
            """),
        md("""
            ## Краткий вывод

            Финальная Leiden-модель дает 20 кластеров и умеренные, но честные
            метрики: ARI около 0.277, V-measure около 0.387, точная pairwise F1
            около 0.339. Это ожидаемо для коротких юмористических текстов:
            один анекдот часто совмещает несколько тем, а теги сайта являются
            шумной внешней разметкой.

            Supervised и semi-supervised результаты полезны как контрольные
            эксперименты: они показывают, что связь между текстом и тегами есть.
            Но они не заменяют основную кластеризацию без учителя.
            """),
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
    parser.add_argument(
        "--output", default="notebooks/Sophie_анеки_кластеризация_итоговая.ipynb"
    )
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
