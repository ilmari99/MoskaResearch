from collections import deque
import itertools as it
import random
import time
from typing import Iterable
from . import utils

class Card:
    """ A class representing a card.
    This is sort of like a named tuple, but with more freedom to customize
    """
    value : int = None
    suit : str = None
    kopled : bool = False
    score : int = None
    _frozen : bool = False
    
    def __init__(self,value,suit,kopled=False,score=None):
        self.value = value
        self.suit = suit
        self.kopled = kopled
        self.score = score
        self._frozen = True
    
    def __setattr__(self, name, value):
        ## TODO
        if name in ["value","suit"] and self._frozen:
            raise TypeError(f"{name} can not be set, as it is frozen.")
        super.__setattr__(self,name, value)
    
    def __hash__(self):
        """ Only hash by value and suit """
        return hash((self.value,self.suit))
        
    
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
    
    def __eq__(self,other):
        return self.value == other.value and self.suit == other.suit
    

class StandardDeck:
    """ The class representing a standard deck implementation as a deque, to mitigate some risks """
    def __init__(self,shuffle : bool=True, seed=None):
        """Initilize the deck with the combinations of card values and card suits

        Args:
            shuffle (bool, optional): Whether to shuffle the deck. Defaults to True. Else the deck is in the order of a Kartesian product.
        """
        random.seed(seed if seed else random.randint(0,1000000))
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


if __name__ == "__main__":
    start = time.time()
    for i in range(1000):
        deck = StandardDeck(shuffle=False)
    print("Time to init 1000 decks: ",time.time()-start)