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
        self.coefficients = [1 for i in range(self.mat_rank)]
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