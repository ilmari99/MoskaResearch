from __future__ import annotations
from abc import abstractmethod
from collections import Counter, namedtuple
from dataclasses import dataclass
import itertools
import logging
import random
import threading
import time
import warnings
import numpy as np
from typing import Any, Dict, List,TYPE_CHECKING, Tuple
from .AbstractEvaluatorBot import AbstractEvaluatorBot
from ..Game.GameState import FullGameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class AbstractHIFEvaluatorBot(AbstractEvaluatorBot):
    """ This class is an abstract class for bots that evaluate the game states.
    This is a modified version of AbstractEvaluatorBot, which fixes the problem of perfect information.

    This class generates all the reachable next states,
    but if there is hidden information in the next state (other than the other players hands),
    this class will sample the next possible states, evaluate them, and uses the average value of an evaluation.

    """
    def __init__(self, moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=0,
                 requires_graphic: bool = False, 
                 log_level=logging.INFO,
                 log_file="",
                 max_num_states : int = 1000,
                 max_num_samples : int = 100,
                 sampling_bias : float = 0,
                 ):
        self.get_nmoves = True
        self.max_num_samples = max_num_samples
        self.max_num_states = max_num_states
        self.sampling_bias = sampling_bias
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file,max_num_states)
    
    @abstractmethod
    def evaluate_states(self, states : List[FullGameState]) -> List[float]:
        pass

    def _make_mock_move(self,move,args) -> List[FullGameState]:
        """ Make a mock move on the game state, without changing the game state.
        This overwrites the superclass, to return a list of states, because the next state might not be known.
        """
        states = []
        state = self.moskaGame._make_mock_move(move,args)
        if move == "PlayToOther":
            # See which cards were lifted
            curr_cards = self.hand.copy().cards
            lifted_cards = [c for c in state.full_player_cards[self.pid] if c not in curr_cards]
            self.plog.debug(f"Full information lifted cards: {lifted_cards}")
            # If there are no lifted cards, then the next immediate state is known
            if len(lifted_cards) == 0:
                states = [state]
            else:
                #Discard the knowledge of the lifted cards, and create states,
                # where the lift is a random sample of cards possibly in deck
                lifted_card_indices = [i for i,c in enumerate(state.full_player_cards[self.pid]) if c in lifted_cards]
                card_samples = self.moskaGame.card_monitor.get_sample_cards_from_deck(self, len(lifted_cards),self.max_num_samples)
                for cards in card_samples:
                    sample_state = state.copy()
                    for i, index_to_change in enumerate(lifted_card_indices):
                        sample_state.full_player_cards[self.pid][index_to_change] = cards[i]
                    states.append(sample_state)
        elif move == "InitialPlay":
            curr_cards = self.hand.copy().cards
            lifted_cards = [c for c in state.full_player_cards[self.pid] if c not in curr_cards]
            self.plog.debug(f"Full information lifted cards: {lifted_cards}")
            if len(lifted_cards) == 0:
                states = [state]
            else:
                #Discard the knowledge of the lifted cards, and create states,
                # where the lift is a random sample of cards possibly in deck
                lifted_card_indices = [i for i,c in enumerate(state.full_player_cards[self.pid]) if c in lifted_cards]
                for cards in self.moskaGame.card_monitor.get_sample_cards_from_deck(self, len(lifted_cards),self.max_num_samples):
                    sample_state = state.copy()
                    for i, index_to_change in enumerate(lifted_card_indices):
                        sample_state.full_player_cards[self.pid][index_to_change] = cards[i]
                    states.append(sample_state)

        elif move == "EndTurn":
            self.moskaGame.cards_to_fall.copy(), self.moskaGame.cards_to_fall.copy() + self.moskaGame.fell_cards.copy()
            if args[1] == self.moskaGame.cards_to_fall.copy():
                curr_cards = self.hand.copy().cards + self.moskaGame.cards_to_fall.copy()
            elif args[1] == self.moskaGame.cards_to_fall.copy() + self.moskaGame.fell_cards.copy():
                curr_cards = self.hand.copy().cards + self.moskaGame.cards_to_fall.copy() + self.moskaGame.fell_cards.copy()

            lifted_cards = [c for c in state.full_player_cards[self.pid] if c not in curr_cards]
            self.plog.debug(f"Full information lifted cards: {lifted_cards}")
            if len(lifted_cards) == 0:
                states = [state]
            else:
                lifted_card_indices = [i for i,c in enumerate(state.full_player_cards[self.pid]) if c in lifted_cards]
                for cards in self.moskaGame.card_monitor.get_sample_cards_from_deck(self, len(lifted_cards),self.max_num_samples):
                    sample_state = state.copy()
                    for i, index_to_change in enumerate(lifted_card_indices):
                        sample_state.full_player_cards[self.pid][index_to_change] = cards[i]
                    states.append(sample_state)
        else:
            states = [state]
        return states
    
    def _get_move_prediction(self, move : str, get_n : bool = False) -> Tuple[Any,float]:
        """ Get a prediction for a moves best 'goodness' """
        plays, states, evals = self.get_possible_next_states(move)
        # Here there will be duplicate plays, and corresponding states and evals
        # We need to remove the duplicates, and calculate the mean eval for each unique play
        if len(plays) == 0:
            raise ValueError("No possible next states for move: " + move)
        # If the move is PlayFallFromDeck, there can be uncertainty about the future states (if len(deck) > 1)
        # so we need to calculate the mean evaluation of the possible states
        if move == "PlayFallFromDeck":
            # Store the scores, to be able to get the pre-computed score for a specific play
            self.play_fall_from_deck_scores = {tuple(play) : eval for play, eval in zip(plays, evals)}
            plays = ["unknown"]
            states = ["unknown"]
            evals = [float(np.mean(evals))]

        # In these moves there is also uncertainty if there is deck left, so we need to sample them
        # Each unique play corresponds to multiple possible states.
        # Calculate the mean evaluation of the possible states for each unique play
        elif move in ["PlayToOther", "InitialPlay", "EndTurn"]:
            unique_plays = []
            mean_evals = []
            corresponding_states = []
            # et the unique plays
            for play in plays:
                if play not in unique_plays:
                    unique_plays.append(play)
            # For each unique play, evaluate all possible future states and use the mean of the evaluations
            for unique_play in unique_plays:
                play_evals = [eval for play, eval in zip(plays, evals) if play == unique_play]
                mean_eval = float(np.mean(play_evals))
                #mean_eval += self.sampling_bias*len(self.moskaGame.card_monitor.get_sample_cards_from_deck(self,1,52))
                mean_eval += self.sampling_bias if len(self.moskaGame.deck) > 0 else 0
                mean_evals.append(mean_eval)
                self.plog.debug(f"Sampled {len(evals)} possible states for {unique_play}")
                #corresponding_states.append(states[plays.index(unique_play)])
            self.plog.info(f"Unique plays: {unique_plays[:min(len(unique_plays),10)]}")
            self.plog.info(f"Mean evals: {mean_evals[:min(len(unique_plays),10)]}")
            plays = unique_plays
            #states = corresponding_states
            evals = mean_evals
        if np.isnan(evals).any() or np.isinf(evals).any(): 
            raise Exception("Nan in mean evals!")
        combined = list(zip(plays, evals))
        try:
            best = max(combined, key=lambda x : x[1])
        except:
            print("Combined: ", combined, flush=True)
            print("Evals: ", evals, flush=True)
            print("Plays: ", plays, flush=True)
            raise Exception("Could not find best play")
        
        if get_n:
            return best[0],best[1],len(plays)
        return best[0],best[1]