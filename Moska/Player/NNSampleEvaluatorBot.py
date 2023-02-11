from __future__ import annotations
from collections import namedtuple
import logging
import random
import threading
import numpy as np
from .AbstractEvaluatorBot import AbstractEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState, GameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class NNSampleEvaluatorBot(AbstractEvaluatorBot):
    
    def __init__(self,
                 moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=0,
                 requires_graphic: bool = False, 
                 log_level=logging.INFO,
                 log_file="",
                 max_num_states : int = 1000,
                 pred_format : str ="new",
                 model_id : (str or int) = "all",
                 nsamples : int = 10
                 ):
        self.nsamples = nsamples
        self.pred_format = pred_format
        self.max_num_states = max_num_states
        self.model_id = model_id
        if not name:
            name = "NNEV"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states)

    def evaluate_approx_states(self, states: List[FullGameState]) -> List[float]:
        preds = []
        self.plog.info(self.moskaGame)
        for state in states:
            # How many cards we know from each player
            known_cards_on_other_players = [len([c for c in cards if c.suit != "X"]) for cards in state.known_player_cards]
            # How many cards each player has in hand
            num_cards_on_other_players = [len(cards) for cards in state.full_player_cards]
            # How many cards we DONT know from each player
            missing_cards_on_other_players = [on_player - we_know for on_player, we_know in zip(num_cards_on_other_players, known_cards_on_other_players)]
            self.plog.info(f"Missing cards on players: {missing_cards_on_other_players}")
            self.plog.info(f"Own pid: {self.pid}")
            # Remove self from the list
            total_missing_cards = sum(missing_cards_on_other_players) - missing_cards_on_other_players[self.pid]
            if total_missing_cards == 0:
                preds.append(self.__evaluate_states([state])[0])
                continue
            # Get 10 samples of cards from the deck which could be the missing cards
            possible_fills = self.moskaGame.card_monitor.get_hidden_cards(self)
            possible_fills = [random.sample(possible_fills, total_missing_cards) for _ in range(self.nsamples)]
            sampled_states = []
            for possible_cards in possible_fills:
                start_index = 0
                new_state = state.copy()
                for pid in range(4):
                    if pid == self.pid:
                        continue
                    end_index = start_index + missing_cards_on_other_players[pid]
                    if end_index == total_missing_cards:
                        end_index = None
                    self.plog.info(f"Adding cards {possible_cards[start_index:end_index]} to player {pid}")
                    new_state.known_player_cards[pid] += possible_cards[start_index:end_index]
                    start_index += missing_cards_on_other_players[pid]
                sampled_states.append(new_state)
            sampled_preds = self.__evaluate_states(sampled_states)
            preds.append(np.mean(sampled_preds))
        return preds
        
    def __evaluate_states(self, states: List[FullGameState]) -> List[float]:
        state_vectors = [state.as_perspective_vector(self,fmt=self.pred_format) for state in states]
        preds = self.moskaGame.model_predict(np.array(state_vectors, dtype=np.float32), model_id=self.model_id)
        return preds
    
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        preds = self.evaluate_approx_states(states)
        return preds



    
    
                
        