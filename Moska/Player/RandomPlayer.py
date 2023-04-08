from __future__ import annotations
import logging
import random
from .AbstractPlayer import AbstractPlayer
from typing import Dict, List,TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class RandomPlayer(AbstractPlayer):
    def __init__(self, moskaGame: MoskaGame = None, name: str = "", delay=0, requires_graphic: bool = False, log_level=logging.INFO, log_file=""):
        if not name:
            name = "R-"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)
    
    def choose_move(self, playable: List[str]) -> str:
        return random.choice(playable)
    
    def play_fall_card_from_hand(self) -> Dict[Card, Card]:
        """Select random card-in-hand : card-on-table pairs

        Returns:
            Dict[Card, Card]: _description_
        """
        poss_plays = {}
        for card in self.hand:
            poss_plays[card] = self._map_to_list(card)
        out = {}
        for card, plays in poss_plays.items():
            if plays and random.random() > 0.3:
                play = random.choice(plays)
                if play not in out.values():
                    out[card] = play
        return out
    
    def play_initial(self) -> List[Card]:
        """Play a random single card, to the table on an initialization

        Returns:
            List[Card]: _description_
        """
        return [random.choice(self.hand.cards)]
    
    def play_to_self(self) -> List[Card]:
        playb_vals = self._playable_values_from_hand()
        chand = self.hand.copy()
        play_val = playb_vals.pop()
        return chand.pop_cards(cond=lambda x : x.value == play_val,max_cards=random.randint(0,len(self.hand)))
    
    def play_to_target(self) -> List[Card]:
        playb_vals = self._playable_values_from_hand()
        chand = self.hand.copy()
        play_val = playb_vals.pop()
        return chand.pop_cards(cond=lambda x : x.value == play_val,max_cards=random.randint(0,self._fits_to_table()))
    
    def deck_lift_fall_method(self, deck_card: Card) -> Tuple[Card, Card]:
        return (deck_card, random.choice(self.moskaGame.cards_to_fall))
    
    def end_turn(self) -> List[Card]:
        return self.moskaGame.cards_to_fall
    
    
                
        