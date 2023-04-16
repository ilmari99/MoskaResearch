from __future__ import annotations
from collections import namedtuple
import logging
import numpy as np
from .AbstractHIFEvaluatorBot import AbstractHIFEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState
import math
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
                 model_id : (str or int) = "all",
                 sampling_bias : float = 0,
                 min_player : str = "",
                 ):
        self.min_player = min_player
        self.pred_format = pred_format
        self.max_num_states = max_num_states
        self.model_id = model_id
        self.checked_min_pl_exists = False
        if not name:
            name = "NNEVHIF"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states, max_num_samples, sampling_bias)
    
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        if self.min_player:
            if not self.checked_min_pl_exists:
                min_pl = self.moskaGame.get_players_condition(lambda pl: pl.name == self.min_player)
                if not min_pl:
                    raise ValueError(f"Player {self.min_player} does not exist")
                if min_pl[0] is self:
                    raise NameError(f"The player to minimize cant be self.")
                self.checked_min_pl_exists = True
                self.plog.info(f"Minimizing player '{self.min_player}' with pid {min_pl[0].pid}.")
            return self.evaluate_states_min_player_hif(states)
        else:
            return self.evaluate_states_hif(states)
    
    def evaluate_states_hif(self, states: List[FullGameState]) -> List[float]:
        state_vectors = [state.as_perspective_vector(self,fmt=self.pred_format) for state in states]
        states = np.array(state_vectors, dtype=np.float32)
        preds = self.moskaGame.model_predict(states, model_id=self.model_id).flatten().tolist()
        return preds
    
    def evaluate_states_min_player_hif(self, states: List[FullGameState]) -> List[float]:
        raise DeprecationWarning(f"Minimizing a certain player is deprecated for now!!!")
        state_collections = []
        # Get each players state as a perspective vector
        for state in states:
            for pl in self.moskaGame.players:
                state_collections.append(state.as_perspective_vector(pl,fmt=self.pred_format))
        state_collections = np.array(state_collections, dtype=np.float32)
        predictions = self.moskaGame.model_predict(state_collections, model_id=self.model_id).flatten().tolist()
        # Group the predictions by state
        grouped_predictions = [predictions[i:i+len(self.moskaGame.players)] for i in range(0, len(predictions), len(self.moskaGame.players))]
        preds = []
        min_player = self.moskaGame.get_players_condition(lambda pl: pl.name == self.min_player)[0]
        # Use the prediction of all players except the min player as the prediction for the state.
        # This way this player chooses the move, which minimizes every other players chance to lose (maximizes win), except for the min player
        # This basically means that this player plays to only make the min_player lose,
        # and has access to all other players information, except for the min player
        curr_states = [FullGameState.from_game(self.moskaGame, copy=False).as_perspective_vector(pl,fmt=self.pred_format) for pl in self.moskaGame.players]
        curr_evals = self.moskaGame.model_predict(np.array(curr_states,dtype=np.float32),model_id=self.model_id).flatten().tolist()
        def filter_preds(preds):
            for i,p in enumerate(preds):
                if self.moskaGame.players[i].rank is not None:
                    continue
                yield p
        rel_curr_evals = [ev / sum(filter_preds(curr_evals)) for ev in curr_evals]
        for pl_preds in grouped_predictions:
            rel_preds = [ev / sum(filter_preds(pl_preds)) for ev in pl_preds]
            # If MY current evaluation is sufficiently good, or very bad, then only focus on winning.
            # Also, if the min_player's evaluation is already bad enough, then also focus on own game.
            if min_player.rank is None and rel_curr_evals[min_player.pid] > 0.1 and len(self.moskaGame.get_players_condition(lambda x : x.rank is None)) > 3:
                #pred = sum([pr for i, pr in enumerate(rel_preds) if i != min_player.pid])
                pred = 1 - rel_preds[min_player.pid]
                #coeffs = [calc_coeffs(pr, i) for i, pr in enumerate(pl_preds)]
                #pred = sum([coeffs[i] * pl_preds[i] for i,_ in enumerate(pl_preds)])
                #pred = pl_preds[self.pid] - 0.8*pl_preds[min_player.pid]
                #pred = min([pl_pred for i,pl_pred in enumerate(pl_preds) if i != min_player.pid])
                #pred = pl_preds[self.pid] / sum(pl_preds)
            else:
                pred = pl_preds[self.pid]
            preds.append(pred)
        return preds



    
    
                
        