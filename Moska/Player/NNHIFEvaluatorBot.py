from __future__ import annotations
from collections import namedtuple
import logging
import numpy as np
from .AbstractHIFEvaluatorBot import AbstractHIFEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState, GameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class NNHIFEvaluatorBot(AbstractHIFEvaluatorBot):
    
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
                 model_id : (str or int) = "all"
                 ):
        self.pred_format = pred_format
        self.max_num_states = max_num_states
        self.model_id = model_id
        if not name:
            name = "NNEVHIF"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states, max_num_samples)
        
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        state_vectors = [state.as_perspective_vector(self,fmt=self.pred_format) for state in states]
        preds = self.moskaGame.model_predict(np.array(state_vectors, dtype=np.float32), model_id=self.model_id)
        return preds



    
    
                
        