from __future__ import annotations
import random
from typing import List
from .AbstractPlayer import AbstractPlayer
from ..Deck import Card
from .. import utils


class BasePlayer(AbstractPlayer):
    def choose_move(self,playable) -> str:
        """ Select a move to play.

        Args:
            from_ (_type_, optional): _description_. Defaults to None.

        Returns:
            str: _description_
        """
        move = random.choice(playable)
        return move
    
    
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
        
        Default: Play all cards that can be played from hand, to the smallest values in the table.

        Returns:
            _type_: _description_
        """
        play_cards = {}
        for fall_card in self.moskaGame.cards_to_fall:
            for play_card in self.hand:
                success = utils.check_can_fall_card(play_card,fall_card,self.moskaGame.triumph)
                if success:
                    if play_card not in play_cards:
                        play_cards[play_card] = []
                    play_cards[play_card].append(fall_card)
        play_cards = {pc : min(fc) for pc,fc in play_cards.items()}
        pc_inv = {fc : pc for pc,fc in play_cards.items()}
        play_cards = {pc : fc for fc,pc in pc_inv.items()}
        return play_cards
    
    def deck_lift_fall_method(self, deck_card : Card):
        """A function to determine which card will fall, if a random card from the deck is lifted.
        Function should take a card -instance as argument, and return a pair (card_from_deck , card_on_table) in the same order,
        determining which card on the table to fall.
        
        This function is called, when the player decides to koplata and the koplattu card can fall a card on the table.
        If the koplattu card can't fall any card on the table, then the card is just placed on the table.

        Args:
            deck_card (Card): The lifted card from the deck

        Returns:
            tuple(Card,Card): The input card from deck, the card on the table.
        """
        for card in self.moskaGame.cards_to_fall:
            if utils.check_can_fall_card(deck_card,card,self.moskaGame.triumph):
                return (deck_card,card)
            
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
        min_card = min([c.value for c in self.hand])
        hand = self.hand.copy()
        play_cards = hand.pop_cards(cond=lambda x : x.value == min_card,max_cards = self._fits_to_table())
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