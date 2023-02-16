from __future__ import annotations
from collections import namedtuple
import logging
import numpy as np
from .AbstractHIFEvaluatorBot import AbstractHIFEvaluatorBot
from .HeuristicEvaluatorBot import HeuristicEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState, GameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class WideNNEVHEV(AbstractHIFEvaluatorBot):
    
    def __init__(self,
                 moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=0,
                 requires_graphic: bool = False, 
                 log_level=logging.INFO,
                 log_file="",
                 max_num_states : int = 1000,
                 max_num_samples : int = 100,
                 pred_format : str ="new",
                 model_id : (str or int) = "all",
                 hev_coefficients : dict = {},
                 ):
        self.pred_format = pred_format
        self.max_num_states = max_num_states
        self.model_id = model_id
        self.heuristic_evaluator = HeuristicEvaluatorBot(moskaGame, name, delay,
                                                         requires_graphic, log_level, log_file,
                                                         max_num_states, max_num_samples, hev_coefficients)
        if not name:
            name = "WIDE"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states, max_num_samples)

    def normalize_preds(self, preds):
        """ Normalize a list of predictions, so that the values are between 0 and 1"""
        max_pred = max(preds)
        if max_pred == float("inf"):
            max_pred = 1000000
        preds = [pred / max_pred for pred in preds]
        return preds
        
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        """ Evaluate a list of states using a neural network and a heuristic function.
        """
        heuristic_preds = self.heuristic_evaluator.evaluate_states(states)
        norm_heuristic_preds = self.normalize_preds(heuristic_preds)

        state_vectors = [state.as_perspective_vector(self,fmt=self.pred_format) for state in states]
        nn_preds = self.moskaGame.model_predict(np.array(state_vectors, dtype=np.float32), model_id=self.model_id)
        norm_nn_preds = self.normalize_preds(nn_preds)

        preds = [sum(pred) for pred in zip(norm_heuristic_preds, norm_nn_preds)]
        return preds