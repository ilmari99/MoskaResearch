from typing import TYPE_CHECKING, Any, Dict,Callable, List
from abc import ABC,abstractmethod
from ..Deck import Card
from .AbstractPlayer import AbstractPlayer
#from ..Game import MoskaGame

class _Coefficients():
    fall_card : Dict[str,Callable] = {}
    to_target : Dict[str,Callable] = {}
    initial_play : Dict[str,Callable] = {}
    to_self : Dict[str,Callable] = {}
    player : AbstractPlayer = None
    def __init__(self, player : AbstractPlayer) -> None:
        self.player = player
        
    def fall_card_scale_hand_play_score(self,hcard : Card,tcard : Card,**kwargs) -> float:
        """Scales the calculated score for playing hcard to tcard.
        """
        pass
    
    def fall_card_scale_deck_play_score(self, deck_card : Card, tcard : Card) -> float:
        """Scales the calculated score for playing deck_card to tcard
        """
    
    def to_self_scale_play_score(self, card_in_hand : Card, card_to_self : Card):
        """Scales the score calculated for playing card_to_self to self and falling with card_in_hand
        """
        pass
    
    def fall_card_maximum_play_score_from_hand(self, **kwargs):
        """ The largest score to play, when falling card with hand"""
        pass
    
    def to_self_maximum_play_score(self):
        """ The largest score to play, when playing to self and falling a card"""
        pass
    
    def initial_play_scale_score(self,play_cards : List[Card]):
        """ Scale the score of playing play_cards as an initial play"""
        pass
    
class HeuristicCoefficients(_Coefficients):
    method_values = {}
    def __init__(self, player: AbstractPlayer,method_values : Dict[str,float] = {}) -> None:
        super().__init__(player)
        #"""
        self.method_values = {'fall_card_already_played_value': -0.09549183742260889, 
         'fall_card_same_value_already_in_hand': 0.30900799478152363, 
         'fall_card_card_is_preventing_kopling': -0.19098300562505258, 
         'fall_card_deck_card_not_played_to_unique': 0.5164389337487543, 
         'fall_card_threshold_at_start': 40.12031993619332, 
         'initial_play_quadratic_scaler': 0.09179640442201922
         }
         #"""
        """
        self.method_values = {
            "fall_card_already_played_value" : -0.13,
            "fall_card_same_value_already_in_hand" : 0.13,
            "fall_card_card_is_preventing_kopling" : -0.08,
            "fall_card_deck_card_not_played_to_unique" : 0.23,
            "fall_card_threshold_at_start" : 5.5,
            "initial_play_quadratic_scaler" : 0.18,
            
        }
        #"""
        for met, val in method_values.items():
            self.method_values[met] = val
        return
    
    def fall_card_scale_hand_play_score(self, hcard: Card, tcard: Card, **kwargs) -> float:
        scale = 1
        # If value is already played
        if hcard.value in set([c.value for c in self.player.moskaGame.cards_to_fall]):
            scale += self.method_values["fall_card_already_played_value"]
        # If I already have same values in hand, it is perhaps easier to get rid of the card if lifted -> Increase the score
        if tcard.value in set([c.value for c in self.player.hand.cards]):
            scale += self.method_values["fall_card_same_value_already_in_hand"]
        # If the card has been kopled and is preventing us from kopling again
        if tcard.kopled and len(self.player.moskaGame.deck) > 0:
            scale += self.method_values["fall_card_card_is_preventing_kopling"]
        scale = scale*(hcard.score + tcard.score)/(hcard.score - tcard.score)
        #scale = scale*(hcard.score - tcard.score)/tcard.score
        return scale
    
    def fall_card_scale_deck_play_score(self, deck_card: Card, tcard: Card) -> float:
        scale = 1
        # Check if tcard can be fallen with another card
        mapping = self.player._map_each_to_list()
        can_fall_with_other_cards = False
        for hcard, fall_cards in mapping.items():
            if tcard in fall_cards:
                can_fall_with_other_cards = True
                break
        if can_fall_with_other_cards:
            scale += self.method_values["fall_card_deck_card_not_played_to_unique"]
        scale = scale*(deck_card.score + tcard.score) / (deck_card.score - tcard.score)
        #scale = scale*(hcard.score - tcard.score)/(tcard.score + deck_card.score)
        return scale
    
    def to_self_scale_play_score(self, card_in_hand: Card, card_to_self: Card):
        scale = 1#(card_in_hand.score - card_to_self.score)/card_to_self.score
        scale = scale*(card_in_hand.score + card_to_self.score) / (card_in_hand.score - card_to_self.score)
        return scale
    
    def fall_card_maximum_play_score_from_hand(self, **kwargs):
        threshold = ((self.method_values["fall_card_threshold_at_start"]-52)/52)*len(self.player.moskaGame.deck) + 52
        return threshold
    
    def to_self_maximum_play_score(self):
        threshold = -len(self.player.moskaGame.deck) + 52
        return threshold
    
    def initial_play_scale_score(self, play_cards: List[Card]):
        return self.method_values["initial_play_quadratic_scaler"] * len(play_cards) + 1 - self.method_values["initial_play_quadratic_scaler"]
        
                

class StaticCoefficients(_Coefficients):
    method_values = {}
    def __init__(self, player: AbstractPlayer, method_values : Dict[str,float] = {}) -> None:
        super().__init__(player)
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
    
    