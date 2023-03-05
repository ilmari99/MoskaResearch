from __future__ import annotations
from collections import namedtuple
import logging
import random
import numpy as np
from .AbstractHIFEvaluatorBot import AbstractHIFEvaluatorBot
from .AbstractEvaluatorBot import AbstractEvaluatorBot
from .HeuristicEvaluatorBot import HeuristicEvaluatorBot
from ._ScoreCards import _ScoreCards
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState, GameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class WideEvaluatorBot(AbstractEvaluatorBot):
    
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
                 coefficients : dict = {},
                 ):
        self.pred_format = pred_format
        self.max_num_states = max_num_states
        self.model_id = model_id
        self.scorer : _ScoreCards = _ScoreCards(self,default_method="counter")
        self.coefficients = {
            "my_cards" : 6.154,
            "len_set_my_cards" : 2.21,
            "len_my_cards" : 1.57,
            "kopled":-3,
            "missing_card" : 52.6  
        }
        if isinstance(coefficients, str) and coefficients == "random":
            coefficients = {}
            for coef,val in self.coefficients.items():
                coefficients[coef] = val + random.uniform(-val,val)
        for coef, value in coefficients.items():
            if coef not in self.coefficients:
                raise ValueError(f"Unknown coefficient name: {coef}")
            self.coefficients[coef] = value
        if not name:
            name = "WIDE"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states)

    def sigmoid(self, preds):
        return 1 / (1 + np.exp(-0.01*np.array(preds)))
        
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        """ Evaluate a list of states using a neural network and a heuristic function.
        """
        heuristic_preds = self.hev_evaluate_states(states)
        sigmoid_heuristic_preds = self.sigmoid(heuristic_preds)

        state_vectors = [state.as_perspective_vector(self,fmt=self.pred_format) for state in states]
        nn_preds = self.moskaGame.model_predict(np.array(state_vectors, dtype=np.float32), model_id=self.model_id)

        if nn_preds.shape != sigmoid_heuristic_preds.shape:
            try:
                nn_preds = nn_preds.reshape(sigmoid_heuristic_preds.shape)
            except:
                raise ValueError(f"nn_preds.shape != sigmoid_heuristic_preds.shape: {nn_preds.shape} != {sigmoid_heuristic_preds.shape}")
            if nn_preds.shape != sigmoid_heuristic_preds.shape:
                raise ValueError(f"nn_preds.shape != sigmoid_heuristic_preds.shape: {nn_preds.shape} != {sigmoid_heuristic_preds.shape}")
        preds = np.add(nn_preds, sigmoid_heuristic_preds)
        return preds



    def _get_cards_possibly_in_deck_state(self, state : FullGameState) -> List[Card]:
        """ Get cards that are possibly in the deck, in this state. """
        # If no cards in deck, return empty list
        if len(state.deck) <= 0:
            return []
        # If only one card in deck, it is the triumph card, so we know the card
        if len(state.deck) == 1:
            triumph_card = state.deck.cards.copy().pop()
            triumph_card.score = len(state.cards_fall_dict[triumph_card])
            return [triumph_card]
        
        # Cards in self hand, cards in table and known cards are not in the deck
        cards_not_in_deck = state.full_player_cards[self.pid] + state.fell_cards + state.cards_to_fall
        # Add known player cards
        for player_pid, known_cards in enumerate(state.known_player_cards):
            if player_pid != self.pid:
                cards_not_in_deck += [card for card in known_cards if card.suit != "X"]
        cards_possibly_in_deck = set(state.cards_fall_dict.keys()).difference(cards_not_in_deck)
        return cards_possibly_in_deck
        
    def _calc_expected_value_from_lift(self, state : FullGameState) -> float:
        """ Calculate the expected score of a card that is lifted from the deck.
        Check which cards location we know (Cards in hand + other players known cards).
        >> The remaining cards are either in deck, or in players hands.
        Then calculate the total fall score and divide by the number of cards whose location is not known,
        
        """
        # Cards whose location is not known
        cards_possibly_in_deck = self._get_cards_possibly_in_deck_state(state)
        total_possible_falls = sum((c.score for c in cards_possibly_in_deck))
        if len(cards_possibly_in_deck) == 0:
            return 0
        # Calculate the expected score of a card that is lifted from the deck
        e_lifted = total_possible_falls / len(cards_possibly_in_deck)
        return e_lifted
    
    def _lift_n_from_deck(self, cards : List[Card], state : FullGameState) -> float:
        """ Return how many cards must be lifted from the deck, given current state.
        The player might not have to instantly lift cards from the deck.
        If that is the case, this function corresponds to the number of cards that must be lifted from the deck, if the turn ends now.
        """
        deck = state.deck
        missing = 6 - len(cards)
        liftn = 0
        if missing > 0 and len(deck) > 0:
            liftn = min(missing, len(deck),0)
        return liftn
    
    def hev_evaluate_single_state(self, state: FullGameState) -> float:
        """ Evaluate heuristically, how good a state is for the player.
        The evaluation is a linear combination of the following features:
        - The score/card of the cards in hand 
            - if player is target, we evaluate the hand assuming he lifts the unfallen cards from the table
            - If the player lifts cards from the deck, we add the expected score of the lifted cards
        - The number of cards in the hand
        - The number of unique values in the hand
        - The number of cards that are missing from the hand (after possibly lifting cards from deck, or from the table)
        - Whether there is a kopled card on the table
        """
        # Cards in hand at the state
        my_cards = state.full_player_cards[self.pid]
        # If the player is the target, we evaluate the position assuming he lifts the cards from the table
        my_cards += state.cards_to_fall if self.pid == state.target_pid else []
        my_cards_score = sum((c.score for c in my_cards))
        
        # Calculate the expected score of a card that is lifted from the deck
        expected_score_from_lift = self._calc_expected_value_from_lift(state)
        
        # Calculate the number of cards that must be lifted from the deck
        liftn = self._lift_n_from_deck(my_cards, state)
        
        # Calculate the expected score of the lifted cards
        from_lifted_score = expected_score_from_lift * liftn
        
        # How many cards are missing from hand, after possibly lifting cards from deck or table
        missing_from_hand = max(6 - len(my_cards) - liftn,0)

        # If the player has no cards, the score is infinite -> player doesn't lose
        if len(my_cards) + liftn == 0:
            #return float("inf")
            return 1000000
        
        # Calculate the average score of the cards in hand
        avg_hand_score = (my_cards_score + from_lifted_score) / (len(my_cards) + liftn)
        # Create a linear combination of the different factors using weights from self.coefficients
        score = avg_hand_score * self.coefficients["my_cards"]
        score += self.coefficients["kopled"] if any((c.kopled for c in state.cards_to_fall)) else 0
        score += len(set(my_cards)) * self.coefficients["len_set_my_cards"]
        score += missing_from_hand * self.coefficients["missing_card"]
        score += len(my_cards) * self.coefficients["len_my_cards"]
        return score
    
    def _assign_score_to_state_cards(self, state : FullGameState) -> None:
        """ Assign score to cards in the state, based on how many cards they can fall.
        """
        self.scorer._assign_scores_from_to(list(state.cards_fall_dict.keys()), state.cards_fall_dict)
        self.scorer._assign_scores_from_to(state.fell_cards, state.cards_fall_dict)
        self.scorer._assign_scores_from_to(state.cards_to_fall, state.cards_fall_dict)
        self.scorer._assign_scores_from_to(state.full_player_cards[self.pid], state.cards_fall_dict)
        for pid, known_cards in enumerate(state.known_player_cards):
            self.scorer._assign_scores_from_to(known_cards, state.cards_fall_dict)
        return
        
    def hev_evaluate_states(self, states: List[FullGameState]) -> List[float]:
        preds = []
        for state in states:
            self._assign_score_to_state_cards(state)
            pred = self.hev_evaluate_single_state(state)
            preds.append(pred)
        return preds