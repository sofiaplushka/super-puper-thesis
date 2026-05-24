from thesis_pipeline.text_normalization import count_words, normalize_text, text_hash


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a\n\t b  ") == "a b"


def test_text_hash_is_case_insensitive():
    assert text_hash("Привет") == text_hash(" привет ")


def test_count_words_handles_cyrillic():
    assert count_words("раз два, три") == 3

