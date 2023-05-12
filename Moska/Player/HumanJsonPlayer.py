from __future__ import annotations
from typing import TYPE_CHECKING, List
from .AbstractPlayer import AbstractPlayer
from ..Game.Deck import Card
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame for typechecking
    from ..Game.Game import MoskaGame
import logging

class HumanJsonPlayer(AbstractPlayer):
    def __init__(self, moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=10 ** -6,
                 requires_graphic: bool = True,
                 log_level=logging.INFO,
                 log_file=""):
        if not name:
            name = "Human"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)
        self.move_args : str = ""
    
    def choose_move(self, playable) -> str:
        if len(playable) == 1 and playable[0] == "Skip":
            return "Skip"
        s = self.moskaGame._basic_json_repr()
        s = s.replace("\n", "")
        s = s.replace(" ", "")
        s = s.replace("\t", "")
        print(s)
        self.move_args = ""
        # Input should be formatted as: "play;0,1 2,3" or "play;" or "exit;" or "play;1,2,3,4"
        # Where 'play' is the action string identifier, so one of the playable moves.
        while True:
            inp = input().strip("()\n")
            self.plog.info(f"Input: {inp}")
            action, args = inp.split(";")
            self.move_args = args
            if action not in playable:
                print(f"Incorrect action. Action must be one of: {playable}")
                continue
            if action == "exit":
                self.EXIT_STATUS = 2
                self.plog.info(f"Player chose to exit the game.")
                print(f"Exiting game...")
                return "exit"
            break
        return action
    
    
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
    
    def args_to_indices(self, args: str) -> List[int]:
        """ Convert the move arguments to a list of indices. """
        if not args:
            return []
        return [int(i) for i in args.split(",") if i != ""]
    
    def end_turn(self) -> List[Card]:
        """ End turn, pick all cards or not. """
        if self.move_args == "y":
            return self.moskaGame.cards_to_fall + self.moskaGame.fell_cards
        if self.move_args == "n":
            return self.moskaGame.cards_to_fall
        raise ValueError(f"Invalid input. Input must be one of: ['y', 'n']")
    
    def play_initial(self) -> List[Card]:
        """ Select which cards does the user want to play on an initiating turn """
        indices = self.args_to_indices(self.move_args)
        try:
            return [self.hand.cards[i] for i in indices]
        except IndexError:
            print(f"Selected card doesn't exist (IndexError).")
        return []
    
    def play_to_target(self) -> List[Card]:
        """ Which cards to play to target """
        indices = self.args_to_indices(self.move_args)
        try:
            return [self.hand.cards[i] for i in indices]
        except IndexError:
            print(f"Selected card doesn't exist (IndexError).")
    
    def play_fall_card_from_hand(self) -> dict:
        """ Card-in-hand and card-to-fall pairs. """
        out = {}
        args = self.move_args.replace(" ", ",")
        indices = self.args_to_indices(args)
        for i in range(0, len(indices), 2):
            try:
                out[self.hand.cards[indices[i]]] = self.moskaGame.cards_to_fall[indices[i+1]]
            except IndexError:
                print(f"Selected card doesn't exist (IndexError).")
        return out
    
    def deck_lift_fall_method(self, deck_card: Card) -> tuple:
        """ When playing from deck, choose the card to fall from table """
        print(f"Lifted card from deck: {deck_card.as_str(symbol=False)}")
        # Get as input the index of the card to fall from table
        ind = input()
        if self._check_no_input(ind):
            print(f"No input given.")
            return self.deck_lift_fall_method(deck_card)
        try:
            ind = int(ind)
            return (deck_card, self.moskaGame.cards_to_fall[ind])
        except ValueError:
            print(f"Selected card doesn't exist (ValueError).")
            return self.deck_lift_fall_method(deck_card)
        except IndexError:
            print(f"Selected card doesn't exist (IndexError).")
            return self.deck_lift_fall_method(deck_card)
    
    
    def play_to_self(self) -> List[Card]:
        """ Which cards to play to self """
        indices = self.args_to_indices(self.move_args)
        return [self.hand.cards[i] for i in indices]
