# Macro-tag mapping audit

Observed raw tags audited: **158**.
Raw tags still mapped to `other`: **0**.

The `other` bucket became zero in the processed corpus because the mapping is now exhaustive at the raw-tag level.
This is not an expert gold-standard relabeling of individual jokes. It is a tag-level taxonomy: each observed site tag is assigned to a semantic macro-category before validation metrics are computed.

No individual joke rows were manually relabeled. The audit records raw tag counts and example ids only to make the tag-level decisions reviewable.

This affects ARI, AMI, and V-measure by replacing a large residual label with more specific silver-label categories.
The metrics therefore measure alignment with the constructed macro-tag taxonomy, not agreement with expert topic labels.

## Audit status counts

| audit_status        |   count |
|:--------------------|--------:|
| confident           |     108 |
| broad_category      |      33 |
| formal_or_technical |       9 |
| debatable           |       8 |

## Most common mapped macro-tags by raw-tag vocabulary

| mapped_macro_tag               |   raw_tag_count |
|:-------------------------------|----------------:|
| literature_folklore_characters |              26 |
| politics_power                 |              14 |
| transport                      |              12 |
| it_internet                    |              10 |
| everyday_life                  |               9 |
| media_popculture               |               9 |
| sex_gender                     |               7 |
| finance_economy                |               6 |
| religion_holidays              |               6 |
| health_medicine                |               5 |
| family_relationships           |               5 |
| education_children             |               5 |
| army_police                    |               4 |
| work_professions               |               4 |
| science_space                  |               4 |
| sport                          |               4 |
| animals                        |               4 |
| alcohol_smoking_drugs          |               4 |
| consumer_services              |               3 |
| textual_forms                  |               3 |
| cities_places                  |               3 |
| absurd_philosophy              |               3 |
| law_crime                      |               2 |
| food_diet                      |               2 |
| soviet_history_politics        |               2 |
| covid_pandemic                 |               1 |
| ethnic_regional                |               1 |

## Debatable mappings to review

| raw_tag         |   raw_tag_count | mapped_macro_tag               | audit_rationale                                                                                              | example_joke_ids                                       |
|:----------------|----------------:|:-------------------------------|:-------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------|
| звёзды          |              85 | media_popculture               | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | -1092600011;-1123100005;-10095264;-10080386;-9994286   |
| Рабинович       |              77 | literature_folklore_characters | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | -482400001;14517;-61500004;-112000002;-1072700004      |
| блондинки       |              47 | sex_gender                     | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | 112265;185566;195377;236962;268577                     |
| Ржевский        |              37 | literature_folklore_characters | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | -412300007;-432500003;-451000007;-471700002;-471400003 |
| чапаев          |              35 | literature_folklore_characters | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | -422500001;-452600001;-451100001;-451100003;-451100004 |
| пошлые          |              19 | sex_gender                     | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | 433438;544377;551095;551078;563125                     |
| армянское радио |              18 | literature_folklore_characters | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | 11078;-22200004;-111700002;-1121300004;-2022500002     |
| брюнетки        |               6 | sex_gender                     | A site tag can carry several cultural or stylistic readings; the macro-category is a semantic approximation. | -2062100002;319657;350766;433595;899033                |