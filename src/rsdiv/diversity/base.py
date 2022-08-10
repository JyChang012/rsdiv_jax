from abc import ABC, abstractmethod
from typing import Optional, Sequence

import jax.numpy as jnp


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        quality_scores: jnp.ndarray,
        *,
        similarity_scores: Optional[jnp.ndarray],
        embeddings: Optional[jnp.ndarray],
        k: int,
    ) -> Sequence[int]:
        raise NotImplementedError("Rerank method not implemented!")
