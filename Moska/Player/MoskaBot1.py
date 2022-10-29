from __future__ import annotations
from collections import Counter
import random
from typing import TYPE_CHECKING, List
from ..Deck import Card
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame for typechecking
    from ..Game import MoskaGame
from .AbstractPlayer import AbstractPlayer
import logging


class MoskaBot1(AbstractPlayer):
    def __init__(self,
                 moskaGame: MoskaGame = None, 
                 pid: int = 0, 
                 name: str = "", 
                 delay=10 ** -6, 
                 requires_graphic: bool = False, 
                 debug: bool = False, 
                 log_level=logging.INFO, 
                 log_file=""):
        if not name:
            name = f"B1-{pid}"
        super().__init__(moskaGame, pid, name, delay, requires_graphic, debug, log_level, log_file)
    
    def choose_move(self, playable: List[str]) -> str:
        return random.choice(playable)
    
    def end_turn(self) -> List[Card]:
        """Return which cards you want to pick from the table when finishing your turn.
        Default: pick all cards that cannot be fallen.

        Returns:
            list: List of cards to pick from the table
        """
        pick_cards = self.moskaGame.cards_to_fall
        return pick_cards
    
    def play_fall_card_from_hand(self):
        """Return a dictionary of card_in_hand : card_in_table -pairs, denoting which card is used to fall which card on the table.
        This function is called when the player has decided to play from their hand.
        
        Fall the cards with the lowest score

        Returns:
            _type_: _description_
        """
        
        # NOTE: modifying hand and cards_to_fall. Doesn't matter since were are not removing cards from them.
        self.moskaGame.cards_to_fall = self._assign_scores(self.moskaGame.cards_to_fall)
        self.hand.cards = self._assign_scores(self.hand.cards)
        
        # Needed to refer to self.
        def map_to_list(card):
            return card,self._map_to_list(card)
        
        # Map each card in hand, to a list of cards on the table, that can be fallen
        can_fall = map(map_to_list,self.hand)
        
        # Map each card in hand to the card with the smallest score
        def map_to_card(pc_li):
            li = pc_li[1]
            sm_card = self._get_sm_score_in_list(li)
            if not sm_card:
                return pc_li[0],[]
            return pc_li[0],sm_card
        
        # Create a dict with play_card : smallest fallable card
        # and filter if card doesn't map to any card on the table
        play_cards = {pc:fc for pc,fc in map(map_to_card,can_fall) if fc}
        
        # Create a counter to count how many cards from hand are mapped to each card on the table
        counter = Counter(play_cards.values())
        for pc,fc in play_cards.copy().items():
            # If there are multiple mappings to a card on the table, decrement counter and remove the card that is mapped
            # to the card with multiple values
            if counter[fc] > 1:
                counter[fc] -= 1
                play_cards.pop(pc)
        return play_cards
    
    
    def deck_lift_fall_method(self, deck_card : Card):
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

    def play_initial(self):
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
            hand = self.hand.copy()     # Create a copy of the hand
            play_cards = hand.pop_cards(cond=lambda x : x.value in playable_values,max_cards = self._fits_to_table())
        return play_cards
