# Thesis report update guide

## Historical baseline, not final

These values are useful only to describe the development history. Do not insert
them as final thesis results.

- The older untagged corpus had about 8,941 rows. It remains a historical
  reference and is not the final evaluated corpus.
- The first tagged-only pipeline run used 5,509 rows and Leiden clustering over
  BGE-M3/PCA features.
- That initial Leiden run produced 12 clusters.
- Historical initial metrics were moderate: single-label ARI about 0.1332,
  AMI about 0.2625, V-measure about 0.2658, pairwise multilabel F1 about
  0.2589, and silhouette cosine about 0.0366.
- Later exact/remapped old-Leiden baseline values used for before/after
  comparison were ARI 0.1967, V-measure 0.3109, and exact pairwise F1 0.2846.

## Final dataset

The final practical section uses the tagged-only anekdot.ru corpus:

- File: `data/processed/anekdots_tagged.csv`
- Rows: 5,509
- Coverage: 356 of 363 months from 1996-01 through 2026-03
- Every row has at least one raw site tag
- Tags are site-native silver labels, not expert gold labels

This corpus is not directly comparable with the old untagged 8,941-row corpus.
It was selected because tag-based silver labels make external validation
possible without manually labeling individual jokes.

## Final clustering configuration

Use this as the final configuration in the thesis:

- Method: Leiden
- Feature space: hybrid BGE/PCA + lexical TF-IDF/SVD
- Feature set id: `hybrid_dense_lexical_dw0.75_lw0.25`
- Dense weight: 0.75
- Lexical weight: 0.25
- kNN graph: `k=75`, cosine metric
- Leiden resolution: 2.0
- Seed: 7
- Final clusters: 20
- Largest cluster share: about 12.9%

Tags and macro-tags were not appended to text and were not used as clustering
features. They were used only for validation, interpretation, mapping audit,
and configuration selection after clustering.

## Final metrics to report

Final full-dataset metrics:

- ARI: 0.2768
- AMI: 0.3772
- V-measure: 0.3871
- Exact pairwise multilabel precision: 0.3519
- Exact pairwise multilabel recall: 0.3267
- Exact pairwise multilabel F1: 0.3388

Single-clear-label subset:

- V-measure: 0.4198
- Pairwise F1: 0.3541

Before/after against the remapped old-Leiden baseline:

| Metric | Old Leiden | Final | Delta |
|---|---:|---:|---:|
| ARI | 0.1967 | 0.2768 | +0.0802 |
| V-measure | 0.3109 | 0.3871 | +0.0762 |
| Exact pairwise F1 | 0.2846 | 0.3388 | +0.0542 |
| Pairwise precision | 0.2297 | 0.3519 | +0.1221 |
| Largest cluster share | 0.2046 | 0.1291 | -0.0755 |

Final internal metrics are in:

- `outputs/tables/final_internal_cluster_metrics.csv`
- `outputs/tables/internal_cluster_metrics.csv`

The old 12-cluster internal metrics are archived as:

- `outputs/tables/internal_cluster_metrics_initial_leiden.csv`

## What not to overstate

- Do not describe the final result as supervised classification quality.
- Do not claim that clusters fully recover all thematic categories.
- Do not claim that site tags are expert labels.
- Do not imply that UMAP coordinates were used for clustering.
- Do not hide that the final metrics remain moderate.

The correct interpretation is narrower: the final configuration improves
alignment between unsupervised text clusters and external site-tag silver labels,
reduces cluster imbalance, and gives more interpretable report-ready clusters.
Moderate metric values are expected because short humor texts often combine
several themes and the external labels are noisy multi-label site tags.

## Russian thesis-ready wording

### Dataset replacement

В обновленной практической части исходная выборка была заменена на корпус
анекдотов anekdot.ru, для которых на сайте указаны пользовательские теги.
Итоговый корпус содержит 5 509 записей. Такая замена нужна не для увеличения
объема данных, а для появления внешнего источника оценки: теги сайта можно
использовать как слабые, или silver, метки при проверке качества кластеризации.

### 3D UMAP

Кластеризация выполнялась не в координатах UMAP, а в пространстве признаков:
семантических эмбеддингов BGE-M3, PCA-представления и лексических
TF-IDF/SVD-признаков. Двумерная и трехмерная UMAP-визуализации используются
только как проекции для визуального анализа уже полученных кластеров. Поэтому
3D-график следует интерпретировать как иллюстрацию структуры соседства, а не
как самостоятельное доказательство качества кластеризации.

### Validation methodology

Для оценки качества теги сайта были сопоставлены с укрупненными
макро-категориями по их смыслу. Эти категории не добавлялись в текст,
эмбеддинги или признаки кластеризации. Они использовались только после
кластеризации: для расчета ARI, AMI, V-measure и точной pairwise multilabel
F1-метрики, а также для интерпретации кластеров через доминирующие теги.

### Final metrics

Финальная конфигурация использует Leiden-кластеризацию в гибридном текстовом
пространстве BGE/PCA + lexical TF-IDF/SVD. Она дает 20 интерпретируемых
кластеров, а доля крупнейшего кластера составляет около 12.9%. По сравнению с
ремаппированным исходным Leiden baseline ARI вырос с 0.1967 до 0.2768,
V-measure - с 0.3109 до 0.3871, точная pairwise multilabel F1 - с 0.2846 до
0.3388. Для подмножества анекдотов с одной явной макро-категорией V-measure
достигает 0.4198, а pairwise F1 - 0.3541.

### Why metrics improved

Рост метрик связан с двумя изменениями. Во-первых, была расширена таксономия
макро-тегов: остаточная категория `other` была устранена за счет исчерпывающего
сопоставления наблюдаемых сырых тегов с макро-категориями. Это не является
экспертной построчной разметкой; это проверяемая таксономия на уровне тегов.
Во-вторых, вместо одного плотного embedding-пространства был использован
гибридный набор текстовых признаков, объединяющий семантические BGE/PCA-признаки
и лексические TF-IDF/SVD-признаки. Это улучшило согласование кластеров с
внешними тегами без добавления тегов в признаки.

### Remaining limitations

Даже после улучшения метрики остаются умеренными. Это ожидаемо для корпуса
юмористических текстов: один анекдот часто совмещает несколько тем, а теги
сайта являются шумными silver labels. Поэтому результат следует описывать как
улучшение соответствия между unsupervised-кластерами и внешними тегами, а не
как задачу supervised-классификации с высокой точностью.

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
python scripts/11_compute_final_internal_metrics.py
python scripts/12_macro_tag_mapping_audit.py
python scripts/10_build_execution_summary_notebook.py
python scripts/run_all.py --skip-existing
```
