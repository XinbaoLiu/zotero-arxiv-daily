"""Persistent on-disk cache for corpus (Zotero) embeddings.

Entries are keyed by sha256(model name + encode kwargs + text), so a Zotero item is
re-embedded only when its abstract changes or the model/encode settings change.
The cache file is a plain ``.npz`` (``keys`` + ``emb``) and is restored/saved by
``actions/cache`` in the workflows. Every failure (missing file, corrupt file,
shape mismatch, write error) degrades to "compute everything" and never raises.
"""

import hashlib
import json
import os
import time
from typing import Callable

import numpy as np
from loguru import logger

DEFAULT_CACHE_PATH = ".cache/zotero_embeddings.npz"


def _text_key(prefix: str, text: str) -> str:
    return hashlib.sha256((prefix + "\x00" + (text or "")).encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | None, model: str, encode_kwargs: dict | None = None):
        self.path = path or DEFAULT_CACHE_PATH
        self.prefix = model + "|" + json.dumps(encode_kwargs or {}, sort_keys=True, default=str)
        self.store: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            logger.info(f"Embedding cache not found at {self.path}; all corpus embeddings will be computed.")
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys, emb = data["keys"], data["emb"]
            if len(keys) != len(emb):
                raise ValueError("keys/emb length mismatch")
            self.store = {str(k): emb[i] for i, k in enumerate(keys)}
            logger.info(f"Loaded {len(self.store)} cached embeddings from {self.path}")
        except Exception as exc:
            logger.warning(f"Ignoring unreadable embedding cache {self.path}: {exc}")
            self.store = {}

    def encode(self, texts: list[str], encode_fn: Callable[[list[str]], np.ndarray]) -> np.ndarray:
        keys = [_text_key(self.prefix, t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self.store]
        dims = {v.shape for v in self.store.values()}
        if len(dims) > 1:
            logger.warning("Inconsistent cached embedding shapes; recomputing all.")
            self.store, missing_idx = {}, list(range(len(texts)))
        t0 = time.time()
        if missing_idx:
            # de-duplicate identical texts before encoding
            uniq = {}
            for i in missing_idx:
                uniq.setdefault(keys[i], texts[i])
            new = np.asarray(encode_fn(list(uniq.values())))
            for k, v in zip(uniq.keys(), new):
                self.store[k] = np.asarray(v)
        try:
            out = np.stack([self.store[k] for k in keys]) if keys else np.zeros((0, 0))
        except Exception as exc:  # e.g. shape mismatch between cached and new vectors
            logger.warning(f"Embedding cache inconsistent ({exc}); recomputing all.")
            self.store = {}
            new = np.asarray(encode_fn(texts))
            for k, v in zip(keys, new):
                self.store[k] = np.asarray(v)
            out = new
            missing_idx = list(range(len(texts)))
        logger.info(
            f"Corpus embeddings: {len(texts)} total, {len(texts) - len(missing_idx)} from cache, "
            f"{len(missing_idx)} computed in {time.time() - t0:.1f}s"
        )
        self._save(keys)
        return out

    def _save(self, keep_keys: list[str]) -> None:
        try:
            keep = [k for k in dict.fromkeys(keep_keys) if k in self.store]  # prune to current corpus
            if not keep:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp.npz"
            np.savez(tmp, keys=np.array(keep), emb=np.stack([self.store[k] for k in keep]).astype(np.float32))
            os.replace(tmp, self.path)
            logger.info(f"Saved {len(keep)} embeddings to {self.path}")
        except Exception as exc:
            logger.warning(f"Failed to save embedding cache {self.path}: {exc}")
