from __future__ import annotations
from typing import TYPE_CHECKING, List, Iterable, Callable
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from AbstractPlayer import AbstractPlayer

class _ScoreCards:
    """ Class for assigning scores to cards.
    Each player who uses a scoring system, has a separate instance of this class.
    This class is used to assing scores to cards in the players hand, and in the table,
    to determine which cards are the best to play.
    """
    default_method : Callable = None
    player : AbstractPlayer = None
    methods : dict[str,Callable] = {}
    
    def __init__(self,player : AbstractPlayer,
                 default_method : Callable | str = "basic",
                 ) -> None:
        self.methods = {
            "basic" : self._basic_count_score,
            "counter" : self._count_cards_score,
        }
        # Default method is either a custom callable, or a string which is a key in self.methods
        self.default_method = self.methods[default_method]
        self.methods["default"] = self.default_method
        self.player = player
        
    def _assign_scores_from_to(self, cards : List[Card], card_counter : dict[Card,List[Card]]):
        """ Assign scores to cards in the input list, using the card_counter.
        """
        if self.methods["default"] != self._count_cards_score:
            raise Exception("This method is only for the 'counter' method!")
        for card in cards:
            # Only assign score if card is in card_counter. We cant assign a score to unknown cards in the player hand
            if card in card_counter:
                card.score = len(card_counter[card])
            #card.score = len(card_counter[card])
        return cards
    
    def assign_scores_inplace(self, method : str = "default") -> None:
        """ Assign scores to cards in the players hand and in the table inplace. The method
        is the value at 'default' key in self.methods, which is defined at __init__.
        """
        self.player.moskaGame.cards_to_fall = self._assign_scores(self.player.moskaGame.cards_to_fall,method)
        self.player.hand.cards = self._assign_scores(self.player.hand.cards,method)
        return

    def get_sm_score_in_list(self, cards : List[Card]) -> int:
        """Return the smallest score in the list of cards.
        TODO: This shouldn't be maybe be here.
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
    
    def _count_cards_score(self, card : Card):
        """ Return how many cards can the input card fall. Uses the card_monitor to count the cards."""
        return len(self.player.moskaGame.card_monitor.cards_fall_dict[card])
    
    def _basic_count_score(self,card : Card) -> int:
        """Return how many cards can the input card fall;
        How many cards are smaller and same suit
        or if suit is trump, how many cards are not trump or are smaller trump cards.
        
        Doesn't require counting cards.
        
        Args:
            card (Card): The card for which to count the score

        Returns:
            int: How many cards the card can fall
        """
        if card.suit == self.player.moskaGame.trump:
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
        method = self.methods[method]
        new_cards = []
        for card in cards:
            card.score = method(card)
            new_cards.append(card)
        return new_cards