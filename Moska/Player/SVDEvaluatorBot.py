from __future__ import annotations
import logging
import random
import numpy as np
from .AbstractEvaluatorBot import AbstractEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

# Create a named tuple, to be able to compare assignments.

class SVDEvaluatorBot(AbstractEvaluatorBot):
    def __init__(self,
                 mat_file: str,
                 moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=0,
                 requires_graphic: bool = False, 
                 log_level=logging.INFO,
                 log_file="",
                 max_num_states : int = 1000,
                 coefficients : list[int] = []
                 ):
        if isinstance(coefficients,dict):
            coefficients = list(coefficients.values())
        self.mat = np.load(mat_file,)
        self.mat_rank = self.mat.shape[0]
        self.coefficients = [1 for _ in range(self.mat_rank)]
        # NOT GOOD. Roughly 50% loss percent.
        # Bigger matrix could work better, but SVD is likely not a good choice.
        # Could be useful for dimensionality reduction, in combination with a NN.
        if self.mat_rank == 10:
            default_coefficients = [('0', 21.093483458858955), ('1', 5.172486998831714), ('2', 1.7098063557922385), ('3', 1.3857727602734506), ('4', 3.9157722708119582), ('5', -4.8083871248301335), ('6', -0.9766777121354349), ('7', 1.1210910893597015), ('8', 0.43868362273940176), ('9', -1.3164675154188075)]
            self.coefficients = [c[1] for c in default_coefficients]
        if isinstance(coefficients, str) and coefficients == "random":
            for coef,val in enumerate(self.coefficients):
                self.coefficients[coef] = val + random.uniform(-val,val)
        else:
            for coef, value in enumerate(coefficients):
                self.coefficients[coef] = value
        if not name:
            name = "SVD"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states)
    
    def _evaluate_single_state(self, state: FullGameState) -> float:
        """ Evaluate a single state, using the coefficients and the projection of the state vector to feature space. """
        state_vector = state.as_perspective_vector(self)
        features = np.array(state_vector,dtype=np.float32) @ self.mat.T
        score = np.dot(features, np.array(self.coefficients))
        return score
        
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        preds = []
        for state in states:
            pred = self._evaluate_single_state(state)
            preds.append(pred)
        return preds