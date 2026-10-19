"""Tests for the on-disk corpus embedding cache."""

import numpy as np

from zotero_arxiv_daily.reranker.embedding_cache import EmbeddingCache


def _fn(calls):
    def encode(texts):
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)
    return encode


def test_cache_roundtrip_and_incremental(tmp_path):
    path = str(tmp_path / "c" / "emb.npz")
    calls = []
    a = EmbeddingCache(path, "m", {"task": "x"}).encode(["aa", "bbb"], _fn(calls))
    assert calls == [["aa", "bbb"]] and a.shape == (2, 2)
    calls.clear()
    b = EmbeddingCache(path, "m", {"task": "x"}).encode(["aa", "bbb", "cccc"], _fn(calls))
    assert calls == [["cccc"]]
    np.testing.assert_allclose(b[:2], a)
    calls.clear()
    EmbeddingCache(path, "other-model", {"task": "x"}).encode(["aa"], _fn(calls))
    assert calls == [["aa"]]  # model change invalidates


def test_corrupt_cache_is_ignored(tmp_path):
    path = tmp_path / "emb.npz"
    path.write_bytes(b"garbage")
    calls = []
    out = EmbeddingCache(str(path), "m").encode(["x"], _fn(calls))
    assert calls == [["x"]] and out.shape == (1, 2)
