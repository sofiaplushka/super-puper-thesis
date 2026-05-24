# Audit before strong metric improvement

Dataset rows: **5509**.
Clustered rows: **5509**.
Current cluster count: **12**.

## Current external metrics
|   adjusted_rand_index |   adjusted_mutual_info |   homogeneity |   completeness |   v_measure |   rows |
|----------------------:|-----------------------:|--------------:|---------------:|------------:|-------:|
|              0.133207 |               0.262534 |        0.2854 |       0.248774 |    0.265831 |   5243 |

## Current sampled pairwise metrics
| exact   |   total_possible_pairs |   pairs_evaluated |   true_positive |   false_positive |   false_negative |   true_negative |   precision |   recall |       f1 |
|:--------|-----------------------:|------------------:|----------------:|-----------------:|-----------------:|----------------:|------------:|---------:|---------:|
| False   |               15171786 |           1000000 |           38571 |            77919 |           142941 |          740569 |     0.33111 | 0.212498 | 0.258864 |

## Current top macro-tags
| macro_tag            |   count |     share |
|:---------------------|--------:|----------:|
| other                |    1830 | 0.332184  |
| family_relationships |     777 | 0.141042  |
| everyday_life        |     657 | 0.119259  |
| education_children   |     636 | 0.115447  |
| politics_power       |     607 | 0.110183  |
| work_professions     |     401 | 0.07279   |
| it_internet          |     270 | 0.0490107 |
| character_cycles     |     247 | 0.0448357 |
| army_police          |     245 | 0.0444727 |
| animals              |     111 | 0.0201488 |

## Current top unmapped tags
| tag_raw         |   count |
|:----------------|--------:|
| коронавирус     |     102 |
| Одесса          |      95 |
| звёзды          |      85 |
| секс            |      73 |
| спорт           |      73 |
| авиация         |      68 |
| погода          |      53 |
| объявления      |      51 |
| сигареты        |      48 |
| блондинки       |      47 |
| реклама         |      46 |
| футбол          |      45 |
| отдых           |      45 |
| пенсионеры      |      44 |
| новые русские   |      43 |
| диета           |      40 |
| фильмы          |      39 |
| Ржевский        |      37 |
| Москва          |      37 |
| наркотики       |      33 |
| космос          |      33 |
| аптека          |      32 |
| геи             |      30 |
| поезд           |      29 |
| Шерлок Холмс    |      28 |
| налоги          |      28 |
| смс             |      27 |
| парашют         |      26 |
| приметы         |      26 |
| кризис          |      24 |
| кредиты         |      22 |
| Сочи 2014       |      21 |
| лифт            |      20 |
| пошлые          |      19 |
| тараканы        |      18 |
| армянское радио |      18 |
| золотая рыбка   |      17 |
| метро           |      16 |
| сбербанк        |      16 |
| нанотехнологии  |      16 |

Notebook `notebooks/tagged_corpus_analysis.ipynb`: executed code cells=0, output cells=0.

## Missing artifacts for this goal
- `outputs/tables/metrics_all.csv`
- `outputs/tables/feature_ablation_metrics.csv`
- `outputs/tables/clustering_search_all_runs.csv`
- `notebooks/tagged_corpus_analysis_executed_colab.ipynb`

## Planned commands
```bash
python scripts/06_remap_macro_tags.py
python scripts/07_compute_exact_metrics.py
python scripts/08_feature_ablation_and_search.py
python scripts/09_select_final_and_interpret.py
pytest -q
python scripts/run_all.py --skip-existing
```