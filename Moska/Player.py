from __future__ import annotations
from collections import Counter
import itertools
from typing import TYPE_CHECKING, Iterable, List
from .Deck import Card
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame for typechecking
    from .Game import MoskaGame
from .Hand import MoskaHand
from . import utils
from .BasePlayer import BasePlayer
from .Turns import PlayFallCardFromHand, PlayFallFromDeck, PlayToOther, InitialPlay, EndTurn
import threading
import time
import logging

    
class HumanPlayer(BasePlayer):
    """ Class for letting a human player to play Moska."""
    def __init__(self, moskaGame: MoskaGame, 
                 pid: int = 0, 
                 name: str = "", 
                 delay=1, 
                 requires_graphic : bool = True, 
                 debug=True,
                 log_level=logging.INFO,
                 log_file=""):
        if not name:
            name = f"Human"
        super().__init__(moskaGame, pid, name, delay,requires_graphic,debug=debug,log_level=log_level, log_file=log_file)
        
    def _check_no_input(self,inp) -> bool:
        """Check if the input argument is empty.

        Args:
            inp (input): str or bool

        Returns:
            bool: _description_
        """
        if not inp:
            return True
        if isinstance(inp,list) and inp[0] in ["", " "]:
            return True
        return False
    
    def want_to_fall_cards(self) -> bool:
        """ Ask the user whether they want to fall cards.
        'y' for True, else False
        """
        a = input("Do you want to fall cards from table (y/n):\n")
        return True if a == "y" else False
    
    def end_turn(self) -> List[Card]:
        """ End turn, pick all cards or not. """
        pick_fallen = input("Pick all cards (y/n): ",)
        if pick_fallen == "y":
            return self.moskaGame.cards_to_fall + self.moskaGame.fell_cards
        return self.moskaGame.cards_to_fall
    
    def want_to_end_turn(self) -> bool:
        """ User wants to end turn """
        a = input("End turn (y/n):\n")
        return True if a == "y" else False
    
    def want_to_play_from_deck(self) -> bool:
        """ user wants to play from deck """
        a = input("Do you want to play from deck (y/n):\n")
        return True if a == "y" else False
    
    def play_initial(self) -> List[Card]:
        """ Select which cards does the user want to play on an initiating turn """
        print(self.moskaGame)
        indices = input("Which cards do you want to play (indices of cards in hand separated by space):\n").split(" ")
        if self._check_no_input(indices):
            return []
        indices = [int(d) for d in indices]
        return [self.hand.cards[i] for i in indices]
    
    def play_to_target(self) -> List[Card]:
        """ Which cards to play to target """
        indices = input("Which cards do you want to play (indices of cards in hand separated by space):\n ").split(" ")
        if self._check_no_input(indices):
            return []
        indices = [int(d) for d in indices]
        return [self.hand.cards[i] for i in indices]
    
    def play_fall_card_from_hand(self) -> dict:
        """ Card-in-hand and card-to-fall pairs. """
        pairs = input("Give card pairs; Which cards are used to fall which cards (as index tuples'(a,b)'):\n").strip()
        if self._check_no_input(pairs):
            return {}
        pairs = pairs.split(" ")
        pairs = [p.strip("()") for p in pairs]
        hand_indices = [int(p[0]) for p in pairs]
        table_indices = [int(p[-1]) for p in pairs]
        return {self.hand.cards[ih] : self.moskaGame.cards_to_fall[iff] for ih,iff in zip(hand_indices,table_indices)}
    
    def deck_lift_fall_method(self, deck_card: Card) -> tuple:
        """ When playing from deck, choose the card to fall from table """
        print(f"Card from deck: {deck_card}")
        try:
            fall_index = int(input("Select which card you want to fall from table (index): "))
        except ValueError as ve:
            print(ve)
            return []
        print(f"Card pair: {(deck_card, self.moskaGame.cards_to_fall[fall_index])}")
        return (deck_card, self.moskaGame.cards_to_fall[fall_index])
    
    def want_to_play_to_self(self) -> bool:
        a = input("Do you want to play cards to self (y/n):\n")
        return True if a == "y" else False
    
    def play_to_self(self) -> List[Card]:
        indices = input("Which cards do you want to play to self (indices of cards in hand separated by space):\n ").split(" ")
        if self._check_no_input(indices):
            return []
        indices = [int(d) for d in indices]
        return [self.hand.cards[i] for i in indices]
        

class MoskaBot1(BasePlayer):
    def __init__(self, moskaGame: MoskaGame, 
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
    
    def _count_score(self,card : Card):
        """Return how many cards can the input card fall;
        How many cards are smaller and same suit
        or if suit is triumph, how many cards are not triumph or are smaller triumph cards.

        Args:
            card (Card): The card for which to count the score

        Returns:
            int: How many cards the card can fall
        """
        if card.suit == self.moskaGame.triumph:
            return 4*13 - (14 - card.value)
        else:
            return 12 - (14 - card.value)
    
    def _assign_scores(self, cards : Iterable[Card]) -> List[Card]:
        """Create new Card instances, with the Card instances from Iterable.
        Return the new cards

        Args:
            cards (Iterable[Card]): The cards which are copied to the new list of cards, along with the score
            
        Returns:
            List[Card]: list of the same cards, with a score -attribute
        """
        new_cards = []
        for card in cards:
            if card.score is None:
                card = Card(card.value,card.suit,card.kopled,self._count_score(card))
            new_cards.append(card)
        return new_cards
         
    def want_to_fall_cards(self) -> bool:
        """Whether the player wants to fall cards from their hand.
        
        Default: Whether play_fall_card_from_hand() returns empty dict.
        This might be slow, but it is correct if play_fall_card_from_hand is deterministic.
        
        This method can be overwritten, but if it is incorrect wrt. play_fall_card_from_hand(), leads to undefined behavior

        Returns:
            bool: Whether the player wants to fall cards from hand
        """
        return bool(self.play_fall_card_from_hand()) 
    
    def want_to_play_to_self(self) -> bool:
        """ Whether the player wants to play cards to self.
        
        Default: Whether play_to_self() returns cards.
        Default might be slower, but it is correct if play_to_self() is deterministic.
        
        This method can be overwritten, but if it is incorrect wrt. play_fall_card_from_hand(), leads to undefined behavior
        
        Returns:
            bool: whether the player wants to play cards to self
        """
        return bool(self.play_to_self())
    
    def end_turn(self) -> List[Card]:
        """Return which cards you want to pick from the table when finishing your turn.
        Default: pick all cards that cannot be fallen.

        Returns:
            list: List of cards to pick from the table
        """
        pick_cards = self.moskaGame.cards_to_fall
        return pick_cards
    
    def _map_to_list(self,card : Card, to : Iterable[Card] = None) -> List[Card]:
        """Return a list of Card -instances from to (default self.moskaGame.cards_to_fall),
        that 'card' can fall.

        Args:
            card (Card): The card that is used to fall cards in 'to'
            to (Iterable[Card], optional): Iterable containing Card -instances. Defaults to cards_on_table.

        Returns:
        List[Card]: List of Card -instances
        """
        if not to:
            to = self.moskaGame.cards_to_fall
        out = []
        for c in to:
            if utils.check_can_fall_card(card,c,self.moskaGame.triumph):
                out.append(c)
        return out
    
    def _get_sm_score_in_list(self,cards : List[Card]):
        """Return the first Card with the smallest score in 'cards'.

        Args:
            cards (List[Card]): _description_

        Returns:
            _type_: _description_
        """        
        if not cards:
            return None
        cards = self._assign_scores(cards)
        sm_score = min((c.score for c in cards))
        return list(filter(lambda x : x.score == sm_score,cards))[0]
    
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
            
    def want_to_end_turn(self) -> bool:
        """ Return True if the player (as target) wants to prematurely end the turn by calling the _end_turn() method 
        which lifts the cards specified in end_turn()
        
        Default: Want to end turn, when there are no cards to fall left"""
        if not self.moskaGame.cards_to_fall:
            return True
        return False
            
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
    
    def want_to_play_from_deck(self):
        """ When you are the target, return True if you want to play from deck (koplata) or else False.
        Default: Play from deck if there is only one card left (which is always triumph). This however is not always the best choice."""
        if len(self.moskaGame.deck) == 1:
            return True
        return False
    
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
