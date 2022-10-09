from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
if TYPE_CHECKING:
    from Game import MoskaGame
    from Deck import Card


class MoskaHand:
    """This class represents a players Hand
    """
    cards = []
    moskaGame = None
    def __init__(self,moskaGame : MoskaGame):
        """Initialize a MoskaHand instance. This requires a reference to a MoskaGame instance.
        This immediately draws 6 cards from the deck associated with the MoskaGame -instance.
        Args:
            moskaGame (MoskaGame): _description_
        """
        self.cards = moskaGame.deck.pop_cards(6)
        self.moskaGame = moskaGame
    
    def draw(self,n : int):
        """Draw n cards from the deck associated with the MoskaGame -instance.
        Add the cards to this hand.

        Args:
            n (_type_): _description_
        """
        self.cards += self.moskaGame.deck.pop_cards(n)
        return
    
    def add(self,cards : Iterable[Card]):
        """Add cards from the iterable, to self.cards

        Args:
            cards (Iterable[Card]): Cards to add
        """
        self.cards += [c for c in cards]
        return
    
    def pop_cards(self,cond = lambda x : True, max_cards = float("inf")):
        """ Pop a maximum of 'max_cards' from the hand, that return True when cond is applied to the card.
        Return the values as a list.
        
        This is a dangerous methdod, that might be used incorrectly when subclassing MoskaPlayer.
        
        TODO: Make safer, for ex by prefixing with "_", but adding the same method without prefix to _MoskaHandCopy
        """
        out = []
        for card in self.cards.copy():
            if cond(card):
                out.append(self.cards.pop(self.cards.index(card)))
            if len(out) >= max_cards:
                break
        return out[0:min(max_cards,len(out))]
    
    def __repr__(self) -> str:
        """What to show when printing self

        Returns:
            str: _description_
        """
        s = ""
        s = s.join((c.as_str()+" " for c in self.cards))
        return s
    
    def __iter__(self):
        """Iterate through cards
        """
        return iter(self.cards)
    
    def __len__(self):
        """ How many cards in hand"""
        return len(self.cards)
    
    def copy(self):
        """ Return a copy of the hand"""
        return _MoskaHandCopy([card for card in self.cards],self.moskaGame)
    

class _MoskaHandCopy(MoskaHand):
    def __init__(self, cards, moskaGame):
        self.moskaGame = moskaGame
        self.cards = cards
        
        