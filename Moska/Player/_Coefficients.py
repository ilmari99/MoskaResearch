from typing import TYPE_CHECKING, Any, Dict,Callable, List
from abc import ABC,abstractmethod
if TYPE_CHECKING:
    from .AbstractPlayer import AbstractPlayer


class Coefficients:
    fall_card : Dict[str,Callable] = {}
    to_target : Dict[str,Callable] = {}
    to_self : Dict[str,Callable] = {}
    player : AbstractPlayer = None
    def __init__(self, player : AbstractPlayer) -> None:
        self.player = player
    
    def fall_card_threshold_play_score(self):
        return 0.2 * 5