from __future__ import annotations
from collections import Counter, namedtuple
import logging
import random
import numpy as np
from ._ScoreCards import _ScoreCards
from .AbstractEvaluatorBot import AbstractEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

# Create a named tuple, to be able to compare assignments.

class HeuristicEvaluatorBot(AbstractEvaluatorBot):
    def __init__(self,
                 moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=0,
                 requires_graphic: bool = False, 
                 log_level=logging.INFO,
                 log_file="",
                 max_num_states : int = 1000,
                 coefficients : Dict[str,float] = {}
                 ):
        self.scorer : _ScoreCards = _ScoreCards(self,default_method="counter")
        self.coefficients = {
            "my_cards" : 2.31923085,
            "len_set_my_cards" : 0.76678776,
            "len_my_cards" : -1.99658315,
            "kopled":0.39232656,
            "missing_card" : 51.86727724  
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
            name = "HEV"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states)
    
    def _get_cards_possibly_in_deck(self, state : FullGameState) -> List[Card]:
        """ Get cards that are possibly in the deck, in this state. """
        # If no cards in deck, return empty list
        if len(state.deck) <= 0:
            return []
        # If only one card in deck, it is the trump card, so we know the card
        if len(state.deck) == 1:
            trump_card = state.deck.cards.copy().pop()
            trump_card.score = len(state.cards_fall_dict[trump_card])
            return [trump_card]
        
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
        cards_possibly_in_deck = self._get_cards_possibly_in_deck(state)
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
    
    def _evaluate_single_state(self, state: FullGameState) -> float:
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
            return 10000000
        
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
        
    def evaluate_states(self, states: List[FullGameState]) -> List[float]:
        preds = []
        for state in states:
            self._assign_score_to_state_cards(state)
            pred = float(self._evaluate_single_state(state))
            preds.append(pred)
        return preds