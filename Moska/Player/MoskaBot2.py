from __future__ import annotations
import logging
import random
from typing import TYPE_CHECKING, Dict, List, Tuple
from Moska import utils
from Moska.Deck import Card
if TYPE_CHECKING:
    from Moska.Game import MoskaGame
from .AbstractPlayer import AbstractPlayer
import numpy as np
from scipy.optimize import linear_sum_assignment

class MoskaBot2(AbstractPlayer):
    cost_matrix_max = 10000
    last_play = ""
    def __init__(self, moskaGame: MoskaGame = None, name: str = "", delay=10 ** -6, requires_graphic: bool = False, log_level=logging.INFO, log_file=""):
        if not name:
            name = "B2-"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)
        
    def choose_move(self, playable: List[str]) -> str:
        # This must be played
        if "InitialPlay" in playable:
            return "InitialPlay"
        if "PlayToOther" in playable and not self.play_to_target():
            playable.pop(playable.index("PlayToOther"))
        if "PlayToSelf" in playable and not self.play_to_self():
            playable.pop(playable.index("PlayToSelf"))
        if "PlayFallFromHand" in playable and not self.play_to_self():
            playable.pop(playable.index("PlayFallFromHand"))
        # Now plays only contain plays, that change the status of the game, ie. we actually want to play something
        if len(playable) > 1 and "Skip" in playable:
            playable.pop(playable.index("Skip"))
        if len(playable) > 1 and "EndTurn" in playable:
            playable.pop(playable.index("EndTurn"))
        self.plog.info(f"Want and can plays: {playable}")
        play = random.choice(playable)
        self.last_play = play
        return play
    
    def end_turn(self) -> List[Card]:
        """Return which cards you want to pick from the table when finishing your turn.
        Default: pick all cards that cannot be fallen.

        Returns:
            list: List of cards to pick from the table
        """
        pick_cards = self.moskaGame.cards_to_fall
        return pick_cards
    
    def play_fall_card_from_hand(self) -> Dict[Card, Card]:
        """Return a dictionary of card_in_hand : card_in_table -pairs, denoting which card is used to fall which card on the table.
        This function is called when the player has decided to play from their hand.
        
        Fall the cards with the lowest score

        Returns:
            _type_: _description_
        """
        
        # NOTE: modifying hand and cards_to_fall. Doesn't matter since were are not removing cards from them.
        self.moskaGame.cards_to_fall = self._assign_scores(self.moskaGame.cards_to_fall)
        self.hand.cards = self._assign_scores(self.hand.cards)
        # Create the cost matrix
        C = self._make_cost_matrix(scoring=lambda c1,c2 : abs(c1.score - c2.score))
        self.plog.info(f"Cost matrix:\n {C}")
        
        # Solve a (possibly) rectangular linear sum assignment problem
        # Return a mapping (indices) from hand to table, that is the optimal assignment, ie. minimize the sum of the associated cost with the assignments.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        hand_indices, fall_indices = linear_sum_assignment(C)
        
        # Loop through the optimal indices, and remove assignments with a large
        play_cards = {}
        for hand_ind, table_ind in zip(hand_indices,fall_indices):
            hand_card = self.hand.cards[hand_ind]
            table_card = self.moskaGame.cards_to_fall[table_ind]
            # Discard cards, that are not that good to play, unless there is no deck left
            if C[hand_ind,table_ind] > 14 and len(self.moskaGame.deck) > 0:
                continue
            # Discard cards that are incorrectly mapped (There will be such cards sometimes)
            if utils.check_can_fall_card(hand_card,table_card,self.moskaGame.triumph):
                play_cards[hand_card] = table_card
        self.plog.info(f"Linear sum assignment: {play_cards}")
        return play_cards
    
    def deck_lift_fall_method(self, deck_card: Card) -> Tuple[Card, Card]:
        """A function to determine which card will fall, if a random card from the deck is lifted.
        Function should take a card -instance as argument, and return a pair (card_from_deck , card_on_table) in the same order,
        determining which card on the table to fall.
        
        This function is called, when the player decides to koplata and the koplattu card can fall a card on the table.
        If the koplattu card can't fall any card on the table, then the card is just placed on the table.
        
        This places the deck_card to the card with the smallest score on the table.
        
        Args:
            deck_card (Card): The lifted card from the deck

        Returns:
            tuple(Card,Card): The input card from deck, the card on the table.
        """
        # Get a list of cards that we can fall with the deck_card
        mapping = self._map_to_list(deck_card)
        # Get the card on the table with the smallest score
        sm_card = self._get_sm_score_in_list(mapping)
        return (deck_card,sm_card)
    
    def play_to_self(self) -> List[Card]:
        """Which cards from hand to play to table.
        Default play all playable values, except triumphs

        Returns:
            List[Card]: list of cards played to self
        """
        pv = self._playable_values_from_hand()
        chand = self.hand.copy()
        cards = chand.pop_cards(cond=lambda x : x.value in pv and x.suit != self.moskaGame.triumph)
        return cards
    
    def play_initial(self) -> List[Card]:
        """ Return a list of cards that will be played to target on an initiating turn. AKA playing to an empty table.
        Default: Play all the smallest cards in hand, that fit to table."""
        self.hand.cards = self._assign_scores(self.hand.cards)
        sm_card = min([c.score for c in self.hand])
        hand = self.hand.copy()
        play_cards = hand.pop_cards(cond=lambda x : x.score == sm_card,max_cards = self._fits_to_table())
        return play_cards
    
    def play_to_target(self) -> List[Card]:
        """ Return a list of cards, that will be played to target.
        This function is called, when there are cards on the table, and you can play cards to a target
        
        This method is meant to be overwriteable.
        Default: Play all playable values that fit.
        """
        playable_values = self._playable_values_from_hand()
        play_cards = []
        if playable_values:
            chand = self.hand.copy()
            chand.cards = self._assign_scores(chand.cards)
            play_cards = chand.pop_cards(cond=lambda x : x.value in playable_values and (x.score < 11 or len(self.moskaGame.deck) <= 0), max_cards = self._fits_to_table())
        return play_cards