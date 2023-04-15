from __future__ import annotations
from typing import TYPE_CHECKING, List
from .AbstractPlayer import AbstractPlayer
from ..Game.Deck import Card
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame for typechecking
    from ..Game.Game import MoskaGame
import logging

class HumanPlayer(AbstractPlayer):
    def __init__(self, moskaGame: MoskaGame = None, name: str = "", delay=10 ** -6, requires_graphic: bool = True, log_level=logging.INFO, log_file=""):
        if not name:
            name = "Human-"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)
    
    def choose_move(self, playable) -> str:
        if len(playable) == 1 and playable[0] == "Skip":
            return "Skip"
        print(f"{self.moskaGame}")
        print(f"Your cards: {self.hand}")
        while True:
            for i,k in enumerate(playable):
                print(f"{i+1}. {k}")
            inp = input(f"What do you want to play: ").strip()
            if self._check_no_input(inp):
                print(f"No input given.")
                continue
            try:
                return playable[int(inp)-1]
            except:
                print(f"Incorrect input. Input must be one of: {list(range(1,len(playable)+1))}")
                continue
    
    
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
        pick_fallen = input("Pick all cards (y/n): ",).strip().lower()
        if pick_fallen == "y":
            return self.moskaGame.cards_to_fall + self.moskaGame.fell_cards
        elif pick_fallen == "n":
            return self.moskaGame.cards_to_fall
        else:
            print("Incorrect input. Input must be 'y' or 'n'.")
            return []
    
    def play_initial(self) -> List[Card]:
        """ Select which cards does the user want to play on an initiating turn """
        indices = input("Which cards do you want to play (indices of cards in hand separated by space):\n")
        indices = indices.split(" ")
        if self._check_no_input(indices):
            return []
        try:
            indices = [int(d)-1 for d in indices]
            return [self.hand.cards[i] for i in indices]
        except Exception as e:
            print(e)
            return []
    
    def play_to_target(self) -> List[Card]:
        """ Which cards to play to target """
        indices = input("Which cards do you want to play (indices of cards in hand separated by space):\n ").strip()
        indices = indices.split(" ")
        if self._check_no_input(indices):
            return []
        try:
            indices = [int(d)-1 for d in indices]
            return [self.hand.cards[i] for i in indices]
        except Exception as e:
            print(e)
            return []
    
    def play_fall_card_from_hand(self) -> dict:
        """ Card-in-hand and card-to-fall pairs. """
        pairs = input("Give card pairs; Which cards are used to fall which cards (as index tuples'(a,b)'):\n").strip()
        if self._check_no_input(pairs):
            return {}
        pairs = pairs.split(" ")
        pairs = [p.strip("()") for p in pairs]
        try:
            # [(0,0),(1,1),(2,2)]
            #hand_indices = [int(p[0]) for p in pairs]
            #table_indices = [int(p[-1]) for p in pairs]
            hand_indices = [int(p.split(",")[0])-1 for p in pairs]
            table_indices = [int(p.split(",")[-1])-1 for p in pairs]
            return {self.hand.cards[ih] : self.moskaGame.cards_to_fall[iff] for ih,iff in zip(hand_indices,table_indices)}
        except Exception as e:
            print(e,flush=True)
            return {}
    
    def deck_lift_fall_method(self, deck_card: Card) -> tuple:
        """ When playing from deck, choose the card to fall from table """
        print(f"Card from deck: {deck_card}")
        try:
            fall_index = int(input("Select which card you want to fall from table (index): ")) -1
            print(f"Card pair: {(deck_card, self.moskaGame.cards_to_fall[fall_index])}")
            return (deck_card, self.moskaGame.cards_to_fall[fall_index])
        except Exception as ve:
            print(ve)
            return self.deck_lift_fall_method(deck_card)
    
    
    def play_to_self(self) -> List[Card]:
        indices = input("Which cards do you want to play to self (indices of cards in hand separated by space):\n ").strip()
        indices = indices.split(" ")
        if self._check_no_input(indices):
            return []
        try:
            indices = [int(d) -1 for d in indices]
            return [self.hand.cards[i] for i in indices]
        except Exception as e:
            print(e)
            return []
