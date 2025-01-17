from typing import Optional, Union

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from scipy import sparse as sps

from .base import BaseRecommender


class FMRecommender(BaseRecommender):
    def __init__(
        self,
        interaction: pd.DataFrame,
        items: pd.DataFrame,
        test_size: Union[float, int] = 0.3,
        random_split: bool = True,
        no_components: int = 10,
        item_alpha: float = 0,
        user_alpha: float = 0,
        loss: str = "bpr",
    ) -> None:
        super().__init__(interaction, items, test_size, random_split)
        self.fm = LightFM(
            no_components=no_components,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            loss=loss,
            random_state=42,
        )

    def _fit(self) -> None:
        self.fm.fit(self.train_mat, epochs=30)

    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_features: Optional[sps.csr_matrix] = None,
        item_features: Optional[sps.csr_matrix] = None,
    ) -> np.ndarray:
        prediction: np.ndarray = self.fm.predict(
            user_ids, item_ids, user_features, item_features
        )
        return prediction

    def precision_at_top_k(self, top_k: int = 5) -> float:
        precision: float = precision_at_k(self.fm, self.test_mat, k=top_k).mean()
        return precision
