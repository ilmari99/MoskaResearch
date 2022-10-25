from __future__ import annotations
from typing import TYPE_CHECKING, List
from ..Deck import Card
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame for typechecking
    from ..Game import MoskaGame
from .BasePlayer import BasePlayer
import logging

class HumanPlayer(BasePlayer):
    """ Class for letting a human player to play Moska."""
    def __init__(self,
                 moskaGame: MoskaGame = None, 
                 pid: int = 0, 
                 name: str = "", 
                 delay=1, 
                 requires_graphic : bool = True, 
                 debug=False,
                 log_level=logging.INFO,
                 log_file=""):
        if not name:
            name = f"Human"
        super().__init__(moskaGame, pid, name, delay,requires_graphic,debug=debug,log_level=log_level, log_file=log_file)
    
    
    def choose_move(self, playable) -> str:
        if len(playable) == 1 and playable[0] == "Skip":
            return "Skip"
        while True:
            for i,k in enumerate(playable):
                print(f"{i}. {k}")
            inp = input(f"What do you want to play: ")
            if self._check_no_input(inp):
                print(f"No input given.")
                continue
            try:
                if int(inp) in range(len(playable)):
                    pass
            except:
                print(f"Incorrect input. Input must be one of: {list(range(len(playable)))}")
                continue
            break
        return playable[int(inp)]
    
    
    def _check_no_input(self,inp) -> bool:
        """Check if the input argument is empty.

        Args:
            inp (input): str or bool

        Returns:
            bool: _description_
        """
        if not inp:
            return True
        if isinstance(inp,list) and inp[0] in ["", " "]:
            return True
        return False
      
    
    def end_turn(self) -> List[Card]:
        """ End turn, pick all cards or not. """
        pick_fallen = input("Pick all cards (y/n): ",)
        if pick_fallen == "y":
            return self.moskaGame.cards_to_fall + self.moskaGame.fell_cards
        return self.moskaGame.cards_to_fall
    
    def play_initial(self) -> List[Card]:
        """ Select which cards does the user want to play on an initiating turn """
        print(self.moskaGame)
        indices = input("Which cards do you want to play (indices of cards in hand separated by space):\n").split(" ")
        if self._check_no_input(indices):
            return []
        indices = [int(d) for d in indices]
        return [self.hand.cards[i] for i in indices]
    
    def play_to_target(self) -> List[Card]:
        """ Which cards to play to target """
        indices = input("Which cards do you want to play (indices of cards in hand separated by space):\n ").split(" ")
        if self._check_no_input(indices):
            return []
        indices = [int(d) for d in indices]
        return [self.hand.cards[i] for i in indices]
    
    def play_fall_card_from_hand(self) -> dict:
        """ Card-in-hand and card-to-fall pairs. """
        pairs = input("Give card pairs; Which cards are used to fall which cards (as index tuples'(a,b)'):\n").strip()
        if self._check_no_input(pairs):
            return {}
        pairs = pairs.split(" ")
        pairs = [p.strip("()") for p in pairs]
        hand_indices = [int(p[0]) for p in pairs]
        table_indices = [int(p[-1]) for p in pairs]
        return {self.hand.cards[ih] : self.moskaGame.cards_to_fall[iff] for ih,iff in zip(hand_indices,table_indices)}
    
    def deck_lift_fall_method(self, deck_card: Card) -> tuple:
        """ When playing from deck, choose the card to fall from table """
        print(f"Card from deck: {deck_card}")
        try:
            fall_index = int(input("Select which card you want to fall from table (index): "))
        except ValueError as ve:
            print(ve)
            return []
        print(f"Card pair: {(deck_card, self.moskaGame.cards_to_fall[fall_index])}")
        return (deck_card, self.moskaGame.cards_to_fall[fall_index])
    
    
    def play_to_self(self) -> List[Card]:
        indices = input("Which cards do you want to play to self (indices of cards in hand separated by space):\n ").split(" ")
        if self._check_no_input(indices):
            return []
        indices = [int(d) for d in indices]
        return [self.hand.cards[i] for i in indices]
