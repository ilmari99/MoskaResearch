from __future__ import annotations
from collections import Counter, namedtuple
import logging
import numpy as np
from ._ScoreCards import _ScoreCards
from .AbstractEvaluatorBot import AbstractEvaluatorBot
from typing import Dict, List,TYPE_CHECKING, Tuple
from ..Game.GameState import FullGameState, GameState
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
                 ):
        self.scorer : _ScoreCards = _ScoreCards(self,default_method="counter")
        if not name:
            name = "HEV"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file, max_num_states)
    
    def _get_cards_possibly_in_deck(self, state : FullGameState) -> List[Card]:
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
        cards_possibly_in_deck = self._get_cards_possibly_in_deck(state)
        total_possible_falls = sum((c.score for c in cards_possibly_in_deck))
        if len(cards_possibly_in_deck) == 0:
            return 0
        # Calculate the expected score of a card that is lifted from the deck
        e_lifted = total_possible_falls / len(cards_possibly_in_deck)
        return e_lifted
    
    
    def _adjust_for_missing_cards(self, cards: List[Card], most_falls,lifted = 0) -> float:
        """ Adjust score for missing cards (each card missing from hand is as valuable as the most falling card)
        """
        missing_from_hand = 6 - len(cards) - lifted
        return max(most_falls * missing_from_hand,0)
    
    def _lift_n_from_deck(self, state : FullGameState) -> float:
        """ Return how many cards must be lifted from the deck, given current state.
        The player might not have to instantly lift cards from the deck.
        If that is the case, this function corresponds to the number of cards that must be lifted from the deck, if the turn ends now.
        """
        cards = state.full_player_cards[self.pid]
        deck = state.deck
        missing = 6 - len(cards)
        liftn = 0
        if missing > 0 and len(deck) > 0:
            liftn = min(missing, len(deck))
        return liftn
    
    def _evaluate_single_state(self, state: FullGameState) -> float:
        """ Evaluate heuristically, how good a state is for the player.
        The score of the player is
        """
        # Cards in my hand at the state
        my_cards = state.full_player_cards[self.pid]
        # Calculate the expected score of a card that is lifted from the deck
        expected_score_from_lift = self._calc_expected_value_from_lift(state)
        # Calculate the number of cards that must be lifted from the deck
        liftn = self._lift_n_from_deck(state)
        # How many cards can the best card in game fall
        most_falls = max((c.score for c in state.cards_fall_dict.keys()))
        # Calculate the score of the hand, not accounting for missing cards
        self.plog.debug(f"Cards in hand scores: {[c.score for c in my_cards]}")
        _pure_hand_score = sum((c.score for c in my_cards))
        self.plog.debug(f"Pure hand score: {_pure_hand_score}")
        # How many cards are missing from hand, after possibly lifting cards from deck
        _missing_from_hand = 6 - len(my_cards) - liftn
        # If the player is the target he has to lift cards from the deck after the turn.
        # This is accounted for by subtracting the number of cards that must be lifted from the deck from the number of missing cards.
        # Take missing cards into account. The score of a missing card is as valuable as the most falling card
        _score_from_missing = max(_missing_from_hand*most_falls,0)
        self.plog.debug(f"Score from missing cards: {_score_from_missing}")
        hand_score = _pure_hand_score + _score_from_missing
        score_from_lifted = expected_score_from_lift * liftn
        self.plog.debug(f"Score from lifted score: {score_from_lifted}")
        self.plog.debug("Total cards in hand: " + str(len(my_cards) + liftn))
        # Return the average score/card of the hand
        if len(my_cards) + liftn == 0:
            return float("inf")
        avg_score = (hand_score + score_from_lifted) / (len(my_cards) + liftn)
        return avg_score
    
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
            pred = self._evaluate_single_state(state)
            preds.append(pred)
        return preds