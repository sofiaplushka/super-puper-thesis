from thesis_pipeline.tag_mapping import map_tags, normalize_tag


def test_normalize_tag_is_case_insensitive():
    assert normalize_tag("  Путин ") == "путин"


def test_map_tags_returns_other_for_unmapped():
    mapping = {"politics_power": ["Путин"], "other": ["юмор"]}
    macros, unmapped = map_tags(["unknown"], mapping)
    assert macros == ["other"]
    assert unmapped == ["unknown"]


def test_map_tags_keeps_mapping_order():
    mapping = {"a": ["x"], "b": ["y"]}
    macros, _ = map_tags(["y", "x"], mapping)
    assert macros == ["a", "b"]

