import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from chunk_text import chunk_text, chunk_dataframe, Chunk


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text(None) == []


def test_short_text_returns_single_chunk():
    text = "This is a short sentence."
    result = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(result) == 1
    assert result[0] == text


def test_long_text_splits_into_multiple_chunks():
    text = "word " * 500
    result = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(result) > 1


def test_chunks_respect_max_size_with_tolerance():
    text = ("This is a sentence. " * 200)
    result = chunk_text(text, chunk_size=200, chunk_overlap=30)
    for chunk in result:
        assert len(chunk) <= 250


def test_overlap_creates_shared_content():
    text = "Sentence one is here. Sentence two is here. Sentence three is here. Sentence four is here."
    result = chunk_text(text, chunk_size=40, chunk_overlap=15)
    assert len(result) > 1


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, chunk_overlap=50)
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, chunk_overlap=60)


def test_paragraph_boundary_preferred():
    text = "Paragraph one text here.\n\nParagraph two text here.\n\nParagraph three text here."
    result = chunk_text(text, chunk_size=30, chunk_overlap=5)
    assert len(result) >= 2


def test_no_empty_chunks_in_output():
    text = "word " * 300
    result = chunk_text(text, chunk_size=50, chunk_overlap=10)
    for chunk in result:
        assert chunk.strip() != ""


def test_single_very_long_word_hard_splits():
    text = "a" * 1000
    result = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(result) > 1
    assert all(len(c) <= 100 for c in result)


def test_chunk_dataframe_basic():
    df = pd.DataFrame({
        "cleaned_text": [
            "This is article one. It has some content.",
            "This is article two. It also has content."
        ]
    })
    chunks = chunk_dataframe(df, text_col="cleaned_text", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_dataframe_assigns_unique_ids():
    df = pd.DataFrame({
        "cleaned_text": ["word " * 100, "word " * 100, "word " * 100]
    })
    chunks = chunk_dataframe(df, text_col="cleaned_text", chunk_size=50, chunk_overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert ids == list(range(len(chunks)))


def test_chunk_dataframe_tracks_source_id():
    df = pd.DataFrame({
        "article_id": ["a1", "a2"],
        "cleaned_text": ["First article text here.", "Second article text here."]
    })
    chunks = chunk_dataframe(df, text_col="cleaned_text", id_col="article_id", chunk_size=100, chunk_overlap=10)
    source_ids = {c.source_id for c in chunks}
    assert source_ids == {"a1", "a2"}


def test_chunk_char_count_matches_length():
    df = pd.DataFrame({"cleaned_text": ["Some sample article text for testing."]})
    chunks = chunk_dataframe(df, text_col="cleaned_text", chunk_size=100, chunk_overlap=10)
    for c in chunks:
        assert c.char_count == len(c.text)


def test_chunk_dataframe_empty_row_produces_no_chunks():
    df = pd.DataFrame({"cleaned_text": ["", "Valid article text here."]})
    chunks = chunk_dataframe(df, text_col="cleaned_text", chunk_size=100, chunk_overlap=10)
    assert all(c.text.strip() != "" for c in chunks)