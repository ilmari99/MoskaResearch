from ..AbstractPlayer import AbstractPlayer
from ...Game.Deck import Card

from typing import TYPE_CHECKING, Any, Dict,Callable, List

class StaticParameters():
    method_values = {}
    player : AbstractPlayer = None
    def __init__(self, player: AbstractPlayer, method_values : Dict[str,float] = {}) -> None:
        raise DeprecationWarning("This class is deprecated, use HeuristicParameters instead")
        self.player = player
        self.method_values = {
            "fall_card_scale_hand_play_score" : 1,
            "fall_card_scale_deck_play_score" : 1,
            "to_self_scale_play_score" : 1,
            "fall_card_maximum_play_score_from_hand" : 14,
            "to_self_maximum_play_score":10,
            "initial_play_scale_score" : 1,
        }
        for met, val in method_values.items():
            self.method_values[met] = val
        return
        
    def fall_card_scale_hand_play_score(self, hcard: Card, tcard: Card, **kwargs) -> float:
        return self.method_values["fall_card_scale_hand_play_score"]
    
    def fall_card_scale_deck_play_score(self, deck_card: Card, tcard: Card) -> float:
        return self.method_values["fall_card_scale_deck_play_score"]
    
    def to_self_scale_play_score(self, card_in_hand: Card, card_to_self: Card):
        return self.method_values["to_self_scale_play_score"]
    
    def fall_card_maximum_play_score_from_hand(self, **kwargs):
        return self.method_values["fall_card_maximum_play_score_from_hand"]
    
    def to_self_maximum_play_score(self):
        return self.method_values["to_self_maximum_play_score"]
    
    def initial_play_scale_score(self, play_cards: List[Card]):
        return self.method_values["initial_play_scale_score"]