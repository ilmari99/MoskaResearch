from collections import deque
from dataclasses import dataclass
import itertools as it
import random
from typing import Iterable
from . import utils

@dataclass(frozen=True, eq=True)
class Card:
    """ A class representing a card.
    This is sort of like a named tuple, but with more freedom to customize
    
    This might be subclassed to make creating algorithms easier. Eq. Adding a moska_value -attribute,
    that stores how many cards the card can fall.
    """
    value : int
    suit : str
    kopled : bool = False
    
    def __repr__(self) -> str:
        """ How to represent the card when printing"""
        return str(f"{utils.suit_to_symbol(self.suit)}{self.value}")
    
    def as_str(self,symbol=True):
        """ Return the attributes as a string"""
        if symbol:
            return self.__repr__()
        return str(f"{self.suit}{self.value}")
    
    def __lt__(self,other):
        """ How to compare the card to others"""
        return self.value < other.value
    


# Could probably be converted to a subclass of deque
class StandardDeck:
    """ The class representing a standard deck implementation as a deque, to mitigate some risks """
    locks = {}      # Currently only supports locking one card at the bottom
    # TODO: deprecate locks
    def __init__(self,shuffle : bool=True):
        """Initilize the deck with the combinations of card values and card suits

        Args:
            shuffle (bool, optional): Whether to shuffle the deck. Defaults to True. Else the deck is in the order of a Kartesian product.
        """
        self.cards = deque((Card(v,s) for v,s in it.product(utils.CARD_VALUES,utils.CARD_SUITS)))
        if shuffle:
            self.shuffle()
        return None
    
    def __repr__(self) -> str:
        """ How to represent the StandardDeck -instance"""
        s = ""
        for st in self.cards:
            s += st.as_str() + " "
        return s
    
    def __len__(self) -> None:
        """ The amount of cards in the deck"""
        return len(self.cards)
    
    def shuffle(self) -> None:
        """ Shuffle the deck inplace """
        random.shuffle(self.cards)
        return None
    
    def pop_cards(self,n) -> None:
        """ Pop n cards from the top of the pile.
        If there are less cards in the pile, than was asked, the rest of the cards are returned, or empty"""
        #print("Current deck: ",self)
        if not self.cards or n <= 0:
            return []
        return [self.cards.popleft() for _ in range(min(len(self),n))]
                
    
    def add(self,cards : Iterable[Card],shuffle=True) -> None:
        """Add cards back to the deck

        Args:
            cards (Iterable[Card]): itarable containing cards.
            shuffle (bool, optional): Whether to shuffle the deck. Defaults to True.

        Raises:
            TypeError: If the Itarable doesn't contain Card instances

        Returns:
            _type_:None
        """
        for card in cards:
            if not isinstance(card,Card):
                raise TypeError("The iterable to add to the deck, must be of type Card")
            self.cards.append(card)
        if shuffle:
            self.shuffle()
        return None
    
    def place_to_bottom(self,card : Card) -> None:
        """Place a card at the bottom of the deck

        Args:
            card (Card): Card to insert to the bottom
        """
        self.cards.append(card)
    
    def insert(self,pos,card,lock = False):
        """ Insert a card to a position in the deck.
        TODO: Make deprecated"""
        self.cards.insert(pos,card)
        if lock:
            self.locks[pos] = card
        