from __future__ import annotations
from collections import Counter
import itertools
import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from ..Game.Deck import Card
from ._ScoreCards import _ScoreCards
if TYPE_CHECKING:
    from ..Game.Game import MoskaGame
from .AbstractPlayer import AbstractPlayer
from .PolicyParameters.HeuristicParameters import HeuristicParameters
from ..Game.utils import check_can_fall_card
from ..Game.GameState import FullGameState

class NewRandomPlayer(AbstractPlayer):
    def __init__(self, moskaGame: MoskaGame = None, name: str = "", delay=0, requires_graphic: bool = False, log_level=logging.INFO, log_file=""):
        if not name:
            name = "NR-"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)

    def _play_move(self) -> Tuple[bool,str]:
        """Calls moskaGame to propose a move.
        This is called on each turn from _continuous play.

        First chooses a move with the abstract method 'choose_move(playable)' where playable contains all the allowed moves.

        After the move is selected, the corresponding wrapper method is called,
        and the moskagame's '_make_move' is called with arguments from the wrapper method.

        Returns:
            Tuple[bool,str] : The first value tells whether the move was valid, and the second tells the reason the move wasn't valid IF the move failed.
        """
        success = False
        # Playable moves
        playable = self._playable_moves()
        # Return the move id to play
        move = self.choose_move(playable)
        state = FullGameState.from_game(self.moskaGame, copy=True)
        # Get the function to call, which returns the arguments to pass to the game
        extra_args = self.moves[move]()
        is_eq, msg = state.is_game_equal(self.moskaGame, return_msg=True)
        if not is_eq:
            print(msg)
            raise AssertionError("Game state changed when getting arguments for making move.")
        # Copy lists, so that they are not modified by the game
        extra_args = [arg.copy() if isinstance(arg,list) else arg for arg in extra_args]
        args = [self] + extra_args
        # Call the game to play the move. Catches Assertion (incorrect move) and Type errors
        success, msg  = self.moskaGame._make_move(move,args)
        # If gathering data, save the state vector
        if (success and (move != "Skip" or len(self.state_vectors) == 0)) and self.moskaGame.GATHER_DATA:
            state = FullGameState.from_game(self.moskaGame, copy=False)
            vec = state.as_perspective_vector(self,fmt="bitmap")
            self.state_vectors.append(vec)
        return success, msg
        
    def choose_move(self, playable: List[str]) -> str:
        """ Choose a random move from playable moves."""
        play = random.choice(playable)
        return play
    
    def end_turn(self) -> List[Card]:
        """
        Choose randomly to pick the all the cards, or only the cards_to_fall
        """
        pick_cards = random.choice([self.moskaGame.cards_to_fall.copy(),self.moskaGame.cards_to_fall.copy() + self.moskaGame.fell_cards.copy()])
        return pick_cards
    
    def play_fall_card_from_hand(self) -> Dict[Card, Card]:
        """Return a random choice of cards to play and fall.
        """
        #self.scoring.assign_scores_inplace()
        # Create the cost matrix
        C = self._make_cost_matrix(scoring = lambda ch,ct : 1 if check_can_fall_card(ch,ct,self.moskaGame.triumph) else 0,max_val=0)
        self.plog.info(f"Cost matrix:\n {C}")
        hand_ind, fall_ind = C.nonzero()
        play_cards = {}
        chand = self.hand.copy()
        ctable = self.moskaGame.cards_to_fall.copy()
        for h,f in zip(hand_ind,fall_ind):
            card_from_hand = chand.cards[h]
            card_on_table = ctable[f]
            if card_from_hand in play_cards.keys() or card_on_table in play_cards.values():
                continue
            if random.random() < 0.5:
                play_cards[card_from_hand] = card_on_table
        return play_cards
    
    def deck_lift_fall_method(self, deck_card: Card) -> Tuple[Card, Card]:
        """Return the card and a random card to fall with the card from the deck
        """
        can_fall = self._map_to_list(deck_card)
        card_on_table = random.choice(can_fall)
        return (deck_card,card_on_table)
    
    def play_to_self(self) -> List[Card]:
        """Return a list of random cards to play to self.
        """
        playable_values = self._playable_values_to_table()
        playable_cards = [c for c in self.hand.copy().cards if c.value in playable_values]
        playable_cards = random.sample(playable_cards,random.randint(0,len(playable_cards)))
        return playable_cards
    
    def play_initial(self) -> List[Card]:
        """Return a list of cards to play from hand to an empty table.
        """
        cards = self.hand.copy().cards
        fits = min(self._fits_to_table(), len(cards))
        self.plog.debug(f"{fits} fits to table")
        #plays = itertools.chain.from_iterable((itertools.combinations(cards,i) for i in range(1,fits + 1)))
        play_iterables = []
        counter = Counter([c.value for c in cards])
        # We can play each single card, and all cards that have at least 2 of the same value
        for i in range(1,fits + 1):
            tmp_cards = cards.copy()
            # Filter out cards that have less than 2 of the same value
            if i > 1:
                tmp_cards = [c for c in tmp_cards if counter[c.value] >= 2]
            # Add the i length combinations to the play_iterables
            play_iterables.append(itertools.combinations(tmp_cards,i))
        plays = itertools.chain.from_iterable(play_iterables)
        legal_plays = []
        for play in plays:
            c = Counter([c.value for c in play])
            if (len(play) == 1 or all((count >= 2 for count in c.values()))):
                legal_plays.append(list(play))
        to_play = random.choice(legal_plays)
        return list(to_play)
    
    def play_to_target(self) -> List[Card]:
        """ Return a list of cards, that will be played to target.
        This function is called, when there are cards on the table, and you can play cards to a target
        
        This method is meant to be overwriteable.
        Default: Play all playable values that fit.
        """
        playable_values = self._playable_values_from_hand()
        playable_cards = [c for c in self.hand.copy().cards if c.value in playable_values]
        playable_cards = random.sample(playable_cards,random.randint(0,min(len(playable_cards),self._fits_to_table())))
        return playable_cards