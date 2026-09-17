from knowledge_storm.utils import (
    REDDIT_CONTENT_MAX_CHARS,
    truncate_text_for_embedding,
)


def test_truncate_text_for_embedding_noop_when_short():
    text = "hello world"
    assert truncate_text_for_embedding(text) == text


def test_truncate_text_for_embedding_prefers_paragraph_boundary():
    long = "a" * 100 + "\n\n" + "b" * 200
    out = truncate_text_for_embedding(long, max_chars=120)
    assert len(out) <= 120
    assert out.endswith("[truncated]")


def test_reddit_cap_is_below_embedding_default():
    assert REDDIT_CONTENT_MAX_CHARS < 24_000
