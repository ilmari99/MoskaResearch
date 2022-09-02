from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
if TYPE_CHECKING:
    from Game import MoskaGame


class MoskaHand:
    cards = []
    moskaGame = None
    def __init__(self,moskaGame : MoskaGame):
        self.cards = moskaGame.deck.pop_cards(6)
        self.moskaGame = moskaGame
    
    def draw(self,n):
        """ Draw n cards from the deck"""
        self.cards += self.moskaGame.deck.pop_cards(n)
        return
    
    def add(self,cards : Iterable):
        self.cards += cards
        return
    
    def pop_cards(self,cond = lambda x : True, max_cards = float("inf")):
        """ Pop a maximum of 'max_cards' from the hand, that return True when cond is applied to the card.
        Return the values as a list"""
        out = []
        for card in self.cards.copy():
            if cond(card):
                out.append(self.cards.pop(self.cards.index(card)))
            if len(out) >= max_cards:
                break
        return out[0:min(max_cards,len(out))]
    
    def __repr__(self) -> str:
        s = ""
        s = s.join((c.as_str()+" " for c in self.cards))
        return s
    
    def __iter__(self):
        return iter(self.cards)
    
    def __len__(self):
        return len(self.cards)