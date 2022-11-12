from __future__ import annotations
from typing import TYPE_CHECKING, List, Iterable, Callable
from functools import wraps
if TYPE_CHECKING:
    from ..Deck import Card
    from AbstractPlayer import AbstractPlayer

class _ScoreCards:
    default_method : Callable = None
    player : AbstractPlayer = None
    
    def __init__(self,player : AbstractPlayer,
                 default_method : Callable = None,
                 ) -> None:
        self.default_method = self._basic_count_score if default_method is None else default_method
        self.player = player
    
    def assign_scores_inplace(self) -> Callable:
        self.player.moskaGame.cards_to_fall = self._assign_scores(self.player.moskaGame.cards_to_fall)
        self.player.hand.cards = self._assign_scores(self.player.hand.cards)

    def get_sm_score_in_list(self,cards : List[Card]):
        """Return the first Card with the smallest score in 'cards'.

        Args:
            cards (List[Card]): _description_

        Returns:
            _type_: _description_
        """        
        if not cards:
            return None
        try:
            sm_score = min((c.score for c in cards))
        except Exception as e:
            print(e)
            print(f"Assign scores to cards first")
            raise Exception(e)
        return list(filter(lambda x : x.score == sm_score,cards))[0]
    
    def _basic_count_score(self,card : Card) -> int:
        """Return how many cards can the input card fall;
        How many cards are smaller and same suit
        or if suit is triumph, how many cards are not triumph or are smaller triumph cards.

        Args:
            card (Card): The card for which to count the score

        Returns:
            int: How many cards the card can fall
        """
        if card.suit == self.player.moskaGame.triumph:
            return 4*13 - (14 - card.value)
        else:
            return 12 - (14 - card.value)
    
    def _assign_scores(self, cards : Iterable[Card],method : Callable = "default") -> List[Card]:
        """Create new Card instances, with the Card instances from Iterable.
        Return the new cards

        Args:
            cards (Iterable[Card]): The cards which are copied to the new list of cards, along with the score
            
        Returns:
            List[Card]: list of the same cards, with a score -attribute
        """
        if method == "default":
            try:
                method = self.default_method
            except AttributeError as ae:
                raise AttributeError(f"No default method found! Specify the method to calculate score, or make sure you have a 'default_method' -attribute available, with a Callable value\n. {ae}")
        new_cards = []
        for card in cards:
            card.score = method(card)
            new_cards.append(card)
        return new_cards