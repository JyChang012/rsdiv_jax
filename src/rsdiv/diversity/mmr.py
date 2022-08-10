from typing import Optional, Sequence
from functools import partial

import jax.numpy as jnp
from jax import jit

from .base import BaseReranker


class MaximalMarginalRelevance(BaseReranker):
    def __init__(self, lbd: float):
        self.lbd = lbd

    @partial(jit, static_argnames=['self', 'embeddings', 'k'])
    def rerank(
        self,
        quality_scores: jnp.ndarray,
        *,
        similarity_scores: jnp.ndarray,
        embeddings: None = None,
        k: int,
    ) -> Sequence[int]:
        n = quality_scores.shape[0]
        k = min(k, n)
        new_selection = jnp.argmax(quality_scores)

        # similarity_scores = jnp.delete(similarity_scores, new_selection, axis=1)
        similarity_scores = similarity_scores.at[:, new_selection].set(-jnp.inf)
        for _ in range(k - 1):
            scores = self.lbd * quality_scores - (1. - self.lbd) * jnp.max(similarity_scores, axis=1)
            scores = scores.at[new_selection].set(-jnp.inf)
            new_selection = jnp.append(new_selection, jnp.argmax(scores))
            # similarity_scores = jnp.delete(similarity_scores, new_selection[-1], axis=1)
            similarity_scores = similarity_scores.at[:, new_selection[-1]].set(-jnp.inf)
        return new_selection
