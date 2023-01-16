from __future__ import annotations
from collections import Counter, namedtuple
import itertools
import logging
import random
import threading
import time
import warnings
import numpy as np
from .AbstractEvaluatorBot import AbstractEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState, GameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

# Create a named tuple, to be able to compare assignments.
# The assignments are the same, if the same cards are played to the same cards, regardless of order.
# This can be checked by sorting the indices, and comparing the sorted lists.
Assignment = namedtuple("Assignment", ["inds"])
Assignment.__eq__ = lambda self, other : self.inds.copy().sort() == other.inds.copy().sort()

class NNEvaluatorBot(AbstractEvaluatorBot):
    
    def __init__(self,
                 moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=0,
                 requires_graphic: bool = False, 
                 log_level=logging.INFO,
                 log_file="",
                 max_num_states : int = 1000,
                 normalize: bool = True,
                 pred_format : str ="new",
                 ):
        self.normalize = normalize
        self.pred_format = pred_format
        self.max_num_states = max_num_states
        if not name:
            name = "NNEV"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states)
        
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        state_vectors = [state.as_perspective_vector(self, norm=self.normalize,fmt=self.pred_format) for state in states]
        preds = self.moskaGame.model_predict(np.array(state_vectors, dtype=np.float32))
        return preds



    
    
                
        