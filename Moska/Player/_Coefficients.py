from typing import TYPE_CHECKING, Any, Dict,Callable, List
from abc import ABC,abstractmethod
from .AbstractPlayer import AbstractPlayer


class _Coefficients():
    fall_card : Dict[str,Callable] = {}
    to_target : Dict[str,Callable] = {}
    initial_play : Dict[str,Callable] = {}
    to_self : Dict[str,Callable] = {}
    player : AbstractPlayer = None
    
    def __init__(self, player : AbstractPlayer) -> None:
        self.player = player
        
    def fall_card_threshold_play_score(self,*args):
        pass

    def fall_card_already_in_table(self,*args):
        pass
    
    def fall_card_same_value_already_in_hand(self,*args):
        pass
    
    def fall_card_card_is_preventing_kopling(self,*args):
        pass

    def play_initial_score_adjustment(self,*args):
        pass


class StaticCoefficients(_Coefficients):
    method_values = {}
    def __init__(self, player: AbstractPlayer, method_values : Dict[str,float] = {}) -> None:
        super().__init__(player)
        self.method_values = {
            "fall_card_threshold_play_score" : 14,
            "fall_card_already_in_table" : -0.5,
            "fall_card_same_value_already_in_hand" : 0.4,
            "fall_card_card_is_preventing_kopling" : -0.6,
            "play_initial_score_adjustment" : -1,
        }
        for met, val in method_values.items():
            self.method_values[met] = val
        return
        
    def fall_card_already_in_table(self, *args):
        return self.method_values["fall_card_already_in_table"]
    
    def fall_card_threshold_play_score(self, *args):
        return self.method_values["fall_card_threshold_play_score"]
    
    def fall_card_same_value_already_in_hand(self, *args):
        return self.method_values["fall_card_same_value_already_in_hand"]
    
    def fall_card_card_is_preventing_kopling(self, *args):
        return self.method_values["fall_card_is_preventing_kopling"]
    
    def play_initial_score_adjustment(self, *args):
        return self.method_values["play_initial_score_adjustment"]
    
    