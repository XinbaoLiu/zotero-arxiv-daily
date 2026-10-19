from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
import time
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from .embedding_cache import EmbeddingCache, DEFAULT_CACHE_PATH
@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        t0 = time.time()
        s1_feature = encoder.encode(s1,**encode_kwargs,show_progress_bar=True)
        logger.info(f"Candidate embeddings: {len(s1)} computed in {time.time() - t0:.1f}s")

        def _encode(texts):
            return encoder.encode(texts, **encode_kwargs, show_progress_bar=True)

        t0 = time.time()
        s2_feature = None
        cache_path = self.config.reranker.local.get("cache_path", DEFAULT_CACHE_PATH)
        if cache_path:
            try:
                cache = EmbeddingCache(
                    cache_path,
                    self.config.reranker.local.model,
                    OmegaConf.to_container(encode_kwargs) if isinstance(encode_kwargs, DictConfig) else dict(encode_kwargs),
                )
                s2_feature = cache.encode(list(s2), _encode)
            except Exception as exc:
                logger.warning(f"Embedding cache failed ({exc}); encoding corpus without cache.")
                s2_feature = None
        if s2_feature is None:
            s2_feature = _encode(s2)
        logger.info(f"Corpus embedding step took {time.time() - t0:.1f}s")
        s1_feature = np.asarray(s1_feature, dtype=np.float32)
        s2_feature = np.asarray(s2_feature, dtype=np.float32)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy() if hasattr(sim, 'numpy') else np.asarray(sim)