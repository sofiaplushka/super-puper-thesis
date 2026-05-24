# Thesis report update guide

## What changed

The practical pipeline was rebuilt around a new tagged-only anekdot.ru corpus.
The old `anekdots.csv`, `clustered_anekdots.csv`, `embeddings.npy`, and `ids.npy`
remain in the repository as historical references, but downstream analysis now
uses:

```text
data/processed/anekdots_tagged.csv
```

Final tagged corpus row count: **5,509**.

Coverage: **356 of 363** months from 1996-01 through 2026-03 had at least one
tagged joke in the parsed monthly archive pages. Seven months had zero tagged
jokes in the parsed archive source.

## Why tagged-only

The new practical part needs external labels for cluster validation. anekdot.ru
tags are not expert gold labels, but they provide site-native silver labels.
This makes it possible to evaluate clustering with tag and macro-tag metrics
without manually labeling individual jokes.

## Main generated artifacts

- `data/processed/anekdots_tagged.csv`
- `data/processed/anekdots_tagged_clustered.csv`
- `data/processed/tag_dictionary.csv`
- `data/processed/tag_macro_mapping.csv`
- `data/embeddings/tagged_bge_m3.npy`
- `data/embeddings/tagged_pca128.npy`
- `outputs/figures/umap2d_leiden.html`
- `outputs/figures/umap3d_leiden.html`
- `outputs/figures/pca3d_baseline.html`
- `outputs/figures/cluster_macro_tag_heatmap.html`
- `outputs/tables/*.csv`
- `outputs/report_notes/*.md`

## 3D method wording

Use this wording in the practical section:

> Clustering was performed in the semantic embedding/PCA space, not in UMAP
> coordinates. UMAP-2D and UMAP-3D were used only as visualization layers for
> the same Leiden cluster labels. Therefore, the 3D plot should be interpreted
> as an exploratory projection rather than as a separate clustering method or
> independent proof of cluster quality.

## Cluster validation wording

Use this wording:

> Cluster quality was evaluated with several complementary measures. Site tags
> were mapped to macro-categories and used as external silver labels. For jokes
> with one macro-category, ARI, AMI, homogeneity, completeness, and V-measure
> were computed. For the full multi-label dataset, pairwise precision, recall,
> and F1 measured whether jokes placed in the same cluster shared at least one
> macro-tag. Internal metrics were computed in embedding/PCA space, not UMAP
> space.

Key validation values:

- Leiden clusters: **12**
- Modularity: **0.5358**
- Single-label ARI: **0.1332**
- Single-label AMI: **0.2625**
- Single-label V-measure: **0.2658**
- Pairwise multilabel F1: **0.2589**
- Silhouette cosine: **0.0366**

## Weaknesses to discuss

- The tagged-only corpus is not directly comparable with the old 8,941-row
  untagged corpus.
- anekdot.ru tags are silver labels and may reflect site tagging practices.
- Macro-category mapping is taxonomy-level manual work, not row-level labeling.
- Some macro-tags dominate the corpus; the largest macro category covers about
  one third of rows.
- UMAP visual separation can look stronger or weaker than actual metric quality.
- Multi-label jokes make single-label external metrics incomplete by design.

## Generated table and figure list

Important tables:

- `outputs/tables/dataset_month_coverage.csv`
- `outputs/tables/top_tags.csv`
- `outputs/tables/cluster_sizes.csv`
- `outputs/tables/cluster_summary.csv`
- `outputs/tables/cluster_tag_purity_entropy.csv`
- `outputs/tables/external_metrics_single_label.csv`
- `outputs/tables/pairwise_multilabel_metrics.csv`
- `outputs/tables/internal_cluster_metrics.csv`
- `outputs/tables/leiden_stability_grid.csv`
- `outputs/tables/cluster_interpretability_summary.csv`

Important figures:

- `outputs/figures/umap2d_leiden.html`
- `outputs/figures/umap3d_leiden.html`
- `outputs/figures/pca3d_baseline.html`
- `outputs/figures/cluster_macro_tag_heatmap.html`
- `outputs/figures/year_coverage.html`
- `outputs/figures/month_coverage.html`

## Final improved clustering wording

Use this wording after the initial validation discussion:

> A follow-up clustering search expanded the macro-tag mapping, removed the
> residual `other` macro bucket, and compared dense semantic, lexical TF-IDF/SVD,
> structural, and hybrid text-only feature sets. Tags were used only for
> validation and configuration ranking, not as clustering input. The selected
> final model was Leiden clustering over a hybrid BGE/PCA plus lexical feature
> space with 20 clusters.

Final improved values:

- Final method: **Leiden**, `k=75`, `resolution=2.0`, `seed=7`
- Final feature set: **hybrid_dense_lexical_dw0.75_lw0.25**
- Final clusters: **20**
- Largest cluster share: **0.1291**
- Excluding-other ARI: **0.2768** (`+0.0802` vs remapped old Leiden)
- Excluding-other V-measure: **0.3871** (`+0.0762`)
- Exact pairwise multilabel F1: **0.3388** (`+0.0542`)
- Single-clear-label V-measure: **0.4198**
- Single-clear-label pairwise F1: **0.3541**

Use this limitation wording:

> The search improved all main external metrics and reduced cluster imbalance,
> but it did not fully reach the most ambitious stretch thresholds. This is
> expected for a multi-label humor corpus where site tags are noisy silver
> labels and jokes often mix several themes. The result should be presented as
> an honest improvement in alignment between unsupervised clusters and external
> tags, not as supervised-quality topic classification.

## Russian thesis-ready text

### Replacement of the dataset

В обновленной практической части исходная выборка была заменена на корпус
анекдотов anekdot.ru, для которых на сайте указаны пользовательские теги.
Итоговый корпус содержит 5 509 записей. Такая замена нужна не для увеличения
объема данных, а для появления внешнего источника оценки: теги сайта можно
использовать как слабые, или silver, метки при проверке качества кластеризации.

### Tagged-only limitation

Корпус с тегами не является полной случайной выборкой всех анекдотов сайта.
Он отражает только те публикации, где доступны теги, поэтому результаты нельзя
напрямую переносить на весь архив anekdot.ru. Кроме того, теги не являются
экспертной разметкой: они могут быть неполными, шумными и неодинаково
подробными для разных тем.

### 3D UMAP

Кластеризация выполнялась не в координатах UMAP, а в пространстве признаков:
семантических эмбеддингов BGE-M3, PCA-представления и дополнительных
текстовых признаков. Двумерная и трехмерная UMAP-визуализации используются
только как проекции для визуального анализа уже полученных кластеров. Поэтому
3D-график следует интерпретировать как иллюстрацию структуры соседства, а не
как самостоятельное доказательство качества кластеризации.

### Validation methodology

Для оценки качества теги сайта были вручную сопоставлены с укрупненными
макро-категориями по их смыслу. Эти категории не добавлялись в текст,
эмбеддинги или признаки кластеризации. Они использовались только после
кластеризации: для расчета ARI, AMI, V-measure и точной pairwise multilabel
F1-метрики, а также для интерпретации кластеров через доминирующие теги.

### Final metrics

Финальная конфигурация использует Leiden-кластеризацию в гибридном текстовом
пространстве BGE/PCA + lexical TF-IDF/SVD. Она дает 20 интерпретируемых
кластеров, а доля крупнейшего кластера снижается до 12.9%. По сравнению с
исходным Leiden baseline ARI вырос с 0.1967 до 0.2768, V-measure — с 0.3109
до 0.3871, точная pairwise multilabel F1 — с 0.2846 до 0.3388. Для подмножества
анекдотов с одной явной макро-категорией V-measure достигает 0.4198, а pairwise
F1 — 0.3541.

### Why metrics improved

Рост метрик связан с двумя изменениями. Во-первых, была расширена
таксономия макро-тегов: остаточная категория `other` была устранена за счет
семантического сопоставления всех наблюдаемых тегов с более точными
макро-категориями. Во-вторых, вместо одного плотного embedding-пространства
был использован гибридный набор текстовых признаков, объединяющий
семантические BGE/PCA-признаки и лексические TF-IDF/SVD-признаки. Это улучшило
согласование кластеров с внешними тегами без добавления тегов в признаки.

### Remaining limitations

Даже после улучшения метрики остаются умеренными. Это ожидаемо для корпуса
юмористических текстов: один анекдот часто совмещает несколько тем, а теги
сайта являются шумными silver labels. Поэтому результат следует описывать как
улучшение соответствия между unsupervised-кластерами и внешними тегами, а не
как задачу supervised-классификации с высокой точностью.

### How not to overstate results

Не следует утверждать, что кластеры полностью совпадают с тематическими
категориями или что модель "понимает" все типы юмора. Корректная формулировка:
финальная конфигурация улучшает внешние метрики, уменьшает дисбаланс
кластеров и дает более интерпретируемые группы, но качество ограничено
много-темностью анекдотов и шумностью тегов.

## Checklist of thesis sections to edit

- Dataset description: replace old corpus description with tagged-only corpus.
- Data collection: describe monthly archive parsing and deterministic sampling.
- Embeddings: state BGE-M3, normalized embeddings, PCA-128, Colab T4 run.
- Clustering: state Leiden on PCA/embedding space, not UMAP.
- Visualization: describe 2D and 3D UMAP as projections.
- Evaluation: add tag-based external metrics, multilabel pairwise metrics, and
  internal metrics.
- Limitations: add tagged-only bias, silver labels, macro mapping, and UMAP caveats.

## Reproduce

```bash
python scripts/01_build_tagged_dataset.py
python scripts/02_compute_embeddings.py --mode local
python scripts/03_cluster_and_visualize.py
python scripts/04_validate_clusters_with_tags.py
python scripts/05_analyze_practical_weaknesses.py
python scripts/06_remap_macro_tags.py
python scripts/07_compute_exact_metrics.py
python scripts/08_feature_ablation_and_search.py
python scripts/09_select_final_and_interpret.py
python scripts/run_all.py --skip-existing
```
