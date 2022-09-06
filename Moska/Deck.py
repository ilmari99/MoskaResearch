from collections import deque
from dataclasses import dataclass
import itertools as it
import random
from typing import Iterable
import utils

@dataclass(frozen=True, eq=True)
class Card:
    value : int
    suit : str
    kopled : bool = False
    
    def __repr__(self) -> str:
        return str(f"{utils.suit_to_symbol(self.suit)}{self.value}")
    
    def as_str(self,symbol=True):
        """ Return the attributes as a string"""
        if symbol:
            return self.__repr__()
        return str(f"{self.suit}{self.value}")
    
    def __lt__(self,other):
        return self.value < other.value
    


# Could probably be converted to a subclass of deque
class StandardDeck:
    locks = {}      # Currently only supports locking one card at the bottom
    def __init__(self,shuffle=True):
        """ Initilize the deck with the combinations of card values and card suits"""
        self.cards = deque((Card(v,s) for v,s in it.product(utils.CARD_VALUES,utils.CARD_SUITS)))
        if shuffle:
            self.shuffle()
        return None
    
    def __repr__(self) -> str:
        s = ""
        for st in self.cards:
            s += st.as_str() + " "
        return s
    
    def __len__(self):
        """ The amount of cards in the deck"""
        return len(self.cards)
    
    def shuffle(self):
        """ Shuffle the deck inplace """
        random.shuffle(self.cards)
        return None
    
    def pop_cards(self,n):
        """ Pop n cards from the top of the pile.
        If there are less cards in the pile, than was asked, the rest of the cards are returned"""
        #print("Current deck: ",self)
        if not self.cards or n <= 0:
            return []
        return [self.cards.popleft() for _ in range(min(len(self),n))]
                
    
    def add(self,cards : Iterable[Card],shuffle=True):
        """ Add cards back to the deck. """
        for card in cards:
            if not isinstance(card,Card):
                raise TypeError("The iterable to add to the deck, must be of type Card")
            self.cards.append(card)
        if shuffle:
            self.shuffle()
        return None
    def place_to_bottom(self,card):
        self.cards.append(card)
    
    def insert(self,pos,card,lock = False):
        """ Insert a card to a position in the deck."""
        #assert pos == -1, "Only locking at the bottom of the queue is supported"
        self.cards.insert(pos,card)
        if lock:
            self.locks[pos] = card
        