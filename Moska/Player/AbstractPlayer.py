from __future__ import annotations
import os
import random
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Set, Tuple
from ..Game.GameState import FullGameState
from ..Game.Deck import Card
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame for typechecking
    from ..Game.Game import MoskaGame
from ..Game.Hand import MoskaHand
from ..Game import utils
from .utils import _make_cost_matrix, _map_each_to_list, _map_to_list
import threading
import time
import logging
import sys
import traceback
from abc import ABC, abstractmethod


class AbstractPlayer(ABC):
    """ Abstract class for a player in Moska.
    This class implements the required methods and some utility methods.
    When subclassing, the abstract methods must be implemented to decide actions.
    
    """
    def __init__(self,
                 moskaGame : MoskaGame = None, 
                 name : str = "", 
                 delay=0,
                 requires_graphic : bool = False,
                 log_level = logging.INFO,
                 log_file = "",
                 min_turns : int = 1,
                 ):
            # -1 = Not-running, 0 = Running, 1 = Clean exit, 2 = Error
        self.EXIT_STATUS : int = -1
        self.hand : MoskaHand = None
        self.pid : int = None
        self.moskaGame : MoskaGame = None
        self.rank : int = None
        self.thread : threading.Thread = None
        self.name : str = ""
        self.ready : bool = False
        self.delay : float = 0
        self.requires_graphic : bool = False
        self.plog : logging.Logger = None
        self.log_level = logging.INFO
        self.log_file : str = ""
        self.thread_id : int = None
        self.moves : Dict[str,Callable] = {}
        self.state_vectors = []
        self.min_turns = min_turns
        self.state_vectors = []

        self.moskaGame = moskaGame
        self.log_level = log_level
        self.name = name
        self.log_file = log_file if log_file else os.devnull
        self.delay = delay
        self.requires_graphic = requires_graphic
        self.moves = {
            "EndTurn" : self._end_turn,
            "InitialPlay" : self._play_initial,
            "PlayToOther" : self._play_to_target,
            "PlayToSelf" : self._play_to_self,
            "PlayFallFromHand" : self._play_fall_card_from_hand,
            "PlayFallFromDeck" : self._play_fall_from_deck,
            "Skip" : self._skip_turn,
        }
    
    def _set_plogger(self) -> None:
        """ Sets the logger for this player.
        Can be called explicitly or with self.log_file=....
        NOTE: This must be called AFTER starting the process in which this player is run in.
        Currently this is called in the `_start` method, which is called from Game when the game begins.
        """
        plog = logging.getLogger(self.name + str(random.randint(0,1000000)))    # TODO: This is why the logs might sometimes display multiple games in one file
        plog.setLevel(self.log_level)
        logging.captureWarnings(True)
        fh = logging.FileHandler(self.log_file,mode="w",encoding="utf-8")
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        fh.setFormatter(formatter)
        plog.addHandler(fh)
        self.plog = plog
        #assert self.plog.hasHandlers(), "Logger has no handles"
        #assert not self.plog.disabled, "Logger is disabled"
        self.plog.debug("Logger succesful")
        return
        
    def _set_moskaGame(self) -> None:
        """Sets the moskaGame instance and deals cards from the deck associated with the moskagame
        """
        self.hand = MoskaHand(self.moskaGame)
        return
    
    def __setattr__(self, name : str, value : Any) -> None:
        """Called when setting a variable in this instance.

        Args:
            name (str): value to set
            value (Any): set value to what
        """
        super.__setattr__(self, name, value)
        # If moskaGame is not set in the constructor, it must be set later
        if name == "moskaGame" and value is not None:
            self._set_moskaGame()
    
    def _set_pid_name_logfile(self,pid) -> None:
        """ Set the players pid. The pid is used to identify the player in the game.
        The pid is the index of the player in the players list of the game.
        Set when the thread is started (_start())
        """
        self.pid = pid
        if self.log_file is not os.devnull:
            self.log_file = utils.add_before("(",self.log_file,str(pid))
    
    def _playable_values_to_table(self) -> Set[int]:
        """Return a set of integer values that can be played to the table.
        This equals the set of values, that have been played to the table.

        This is used as a utility method, and to decide which moves are possible.

        Returns:
            set[int]: Which values have been played to the table
        """
        return set([c.value for c in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards])
    
    def _playable_values_from_hand(self) -> Set[int]:
        """Return a set of values, that can be played to target.
        This is the intersection of values between the values in the table, and in the players hand.

        This is used as a utility method, and to decide which moves are possible.

        Returns:
            set: intersection of played values and values in the hand
        """
        return self._playable_values_to_table().intersection([c.value for c in self.hand.cards])
    
    def _fits_to_table(self) -> int:
        """Return the number of cards playable to the active/target player.
        This equals the number of cards in the targets hand,
        minus the number of (unfallen) cards on the table

        This is used as a utility method, and to decide which moves are possible.

        Returns:
            int: How many cards can be played to the target
        """
        target = self.moskaGame.get_target_player()
        return len(target.hand) - len(self.moskaGame.cards_to_fall)
    
    
    def _play_to_target(self) -> List[AbstractPlayer,List[Card]]:
        """ This is a wrapper method around the abstract 'play_to_target'.        
        """
        play_cards = self.play_to_target()
        target = self.moskaGame.get_target_player()
        self.plog.info(f"Playing {play_cards} to {target.name}")
        #self._playToOther(target,play_cards)
        return [target, play_cards]
    
    def _skip_turn(self) -> List:
        return []
        
    def _play_to_self(self) -> List[AbstractPlayer,List[Card]]:
        """ Play cards selected in play_to_self to self
        Wrapper around the abstract method 'play_to_self'.
        """
        play_cards = self.play_to_self()
        self.plog.info(f"Playing {play_cards} to self.")
        #self._playToOther(self,play_cards)
        return [self, play_cards]
    
    
    def _play_initial(self) -> List[AbstractPlayer,List[Card]]:
        """ This function is called, when self is the initiating player, and gets to play to an empty table.

        This is a wrapper around the abstract method 'play_initial()'

        """
        target = self.moskaGame.get_target_player()
        play_cards = self.play_initial()
        #assert play_cards, f"INITIAL PLAY CAN NOT BE EMPTY"
        self.plog.info(f"Playing {play_cards} to {target.name}")
        #self._initialPlay(target,play_cards)
        return [target, play_cards]
    
    def _can_end_turn(self) -> bool:
        """ Return True if the target CAN end their turn now if they want to.
        True, when all the other players are ready and there are cards on the table (initial play has been played).

        This is used to check whether the EndTurn move is legal.
        """
        players_ready = all((pl.ready for pl in self.moskaGame.players if (pl is not self) and (pl.rank is None)))
        cards_in_table = len(self.moskaGame.cards_to_fall) + len(self.moskaGame.fell_cards) > 0
        return players_ready and cards_in_table
    
    def _must_end_turn(self) -> bool:
        """ Returns True if a player MUST end their turn, i.e. there are no aother valid moves.
        Player must end their turn if:
        - There are cards on the table
        - They have no playable cards
        - All players are ready
        - And the player cant play from deck (there is no deck left or there is a kopled card already)
        """
        if self._can_end_turn() and not self._can_fall_cards() and not self._playable_values_from_hand() and (len(self.moskaGame.deck) == 0 or any(c.kopled for c in self.moskaGame.cards_to_fall)):
            return True
        return False
    
    def _can_fall_cards(self) -> bool:
        """ Returns True if the player can fall cards from table with cards in their hand.

        Returns:
            bool: True if player can kill cards from their hand, else False
        """
        # Loop through the hand, and table and see if there is a card that can be used to kill a card on the table.
        for pc in self.hand:
            for fc in self.moskaGame.cards_to_fall:
                 if utils.check_can_fall_card(pc,fc,self.moskaGame.trump):
                     return True
        return False

    def _play_fall_from_deck(self) -> None:
        """ This is a wrapper method around 'deck_lift_fall_method'
        The 'deck_lift_fall_method' is only called, if the card lifted from deck can fall a card on the table.
        If it can't the card is automatically added to the table.
        """
        return [self.deck_lift_fall_method]

    def _play_fall_card_from_hand(self) -> None:
        """ This is a wrapper around the abstract method 'play_fall_card_from_hand'.
        """
        play_cards = self.play_fall_card_from_hand()
        self.plog.info(f"Falling cards: {play_cards}")
        #self._playFallCardFromHand(play_cards)
        return [play_cards]
    
    def _end_turn(self) -> List[List[Card]]:
        """ Wrapper around 'end_turn'.
        Called when the player must or wants to and can end their turn, or when finishing the game

        Returns:
            bool: True if cards were picked, false otherwise
        """
        pick_cards = []
        # If the player didn't finish fully, ask which cards to pick to hand
        if self.rank is None:
            pick_cards = self.end_turn()
        self.plog.info(f"Ending turn and picking {pick_cards}")
        return [pick_cards]
        
    def _set_rank(self) -> int:
        """Set the players rank. Rank is None, as long as the player is still in the game.
        This is called after each turn in _continuous_play'.
        """
        poss_rank = len(self.moskaGame.get_players_condition(cond = lambda x : x.rank is not None)) + 1
        # If there is only one player left, set rank
        if len(self.moskaGame.get_players_condition(cond = lambda x : x.rank is None)) <= 1:
            self.rank = poss_rank
            self.EXIT_STATUS = 1
        # If the player is not the target, they are finished if:
        # - They have no cards in hand
        # - There are no cards left in the deck
        if self is not self.moskaGame.get_target_player() and (not self.hand and len(self.moskaGame.deck) == 0):
            self.rank = poss_rank
            self.EXIT_STATUS = 1
        # If the player is the target, they are finished if:
        # - They have no cards in hand
        # - There are no cards left in the deck
        # - There are no un-fallen cards on the table
        if self is self.moskaGame.get_target_player() and (not self.hand and len(self.moskaGame.deck) == 0 and not self.moskaGame.cards_to_fall):
            self.rank = poss_rank
            self.EXIT_STATUS = 1
        self.plog.debug(f"Set rank to {self.rank}")
        return self.rank
    
    # Disgusting!
    #@utils.check_new_card
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
        # Get the function to call, which returns the arguments to pass to the game
        extra_args = self.moves[move]()
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
    
    def _playable_moves(self) -> List[str]:
        """ Return the playable moves as a list of move names, such as "EndTurn", "PlayFallFromHand", etc.

        Returns:
            list[str]: List of playable move identifiers
        """
        playable = list(self.moves.keys())
        # If the player has already played the desired cards, and he is not the target
        # If the player is the target, he might not want to play all cards at one turn, since others can then put same value cards to the table
        self.ready = True
        self.plog.debug(f"Player set to ready: {self.ready}")
        # If there are cards on the table; the game is already initiated
        initiated = int(len(self.moskaGame.cards_to_fall) + len(self.moskaGame.fell_cards)) != 0
        # Special case: if the player has played all their cards in the previous turn, they must now end the turn and finish
        if self.rank is not None:
            if self is self.moskaGame.get_target_player():
                playable = ["EndTurn"]
            else:
                playable = ["Skip"]
        # If player is the target
        elif self is self.moskaGame.get_target_player():
            # If the player is the target, they cant play these
            playable.remove("PlayToOther")
            playable.remove("InitialPlay")
            # If the player can not end their turn, they cant end the turn, unless all other players are ready and there are cards played
            if not self._can_end_turn():
                playable.remove("EndTurn")
            # If there are not values to play to self
            if not self._playable_values_from_hand() or len(self.moskaGame.deck) == 0:
                playable.remove("PlayToSelf")
            # If there are no cards to play from hand
            if not self._can_fall_cards():
                playable.remove("PlayFallFromHand")
            # If there is no deck left, or there is already a kopled card on the table, or there are no cards to fall
            if any((c.kopled for c in self.moskaGame.cards_to_fall)) or len(self.moskaGame.deck) <= 0 or not self.moskaGame.cards_to_fall:
                playable.remove("PlayFallFromDeck")
            # If all players are ready and there are no other moves left OR all other players are ready and there are played cards
            if self._must_end_turn() or self._can_end_turn():
                playable.remove("Skip")
                #assert len(playable) == 1, f"There should only be 'end turn' option left. Left options: {playable}"
        else:
            # If the player is not the target player
            playable.remove("PlayFallFromDeck")
            playable.remove("PlayFallFromHand")
            playable.remove("EndTurn")
            playable.remove("PlayToSelf")
            # If the player doesn't have cards to play from hand, or the table is full
            if not self._playable_values_from_hand() or self._fits_to_table() <= 0:
                playable.remove("PlayToOther")
            # If the player is the initiating player and the game is not initiated, they cant skip
            if self is self.moskaGame.get_initiating_player() and not initiated:
                playable.remove("Skip")
            # If the game is initiated, or the player isn't the initiating player, they cant initiate the turn
            if initiated or not self is self.moskaGame.get_initiating_player():
                playable.remove("InitialPlay")
        assert bool(playable), f"There must be something to play"
        self.plog.info(f"Playable moves: {playable}")
        return playable
    
    def _start(self) -> int:
        """ Initializes the players thread, starts the thread and returns the threads native identification """
        self._set_pid_name_logfile(self.moskaGame.players.index(self))
        if self.thread is None or not self.thread.is_alive():
            self._set_plogger()
            # Make daemon, to kill the thread if the mian thread (game) fails in anyway.
            self.thread = threading.Thread(target=self._continuous_play,name=self.name,daemon=True)
            self.plog.info("Initialized thread")
            self.thread.start()
            self.thread_id = self.thread.native_id
        # Means running
        self.EXIT_STATUS = 0
        return self.thread_id
    
    def _continuous_play(self) -> None:
        """ The main method of MoskaPlayer. This is the target of the players thread.
        This method is meant to be run indirectly, by starting the Thread associated with the player.
        This function starts a while loop, that runs as long as the players rank is None and there are atleast 2 players in the game.

        The thread is killed, if the main (game) thread fails for any reason.
        """
        tb_info = {"players" : len(self.moskaGame.players),
                   "Trump card" : self.moskaGame.trump_card,
                   }
        self.plog.info(f"Table info: {tb_info}")
        curr_target = self.moskaGame.get_target_player()
        turns_taken_for_this_player = 0
        self.rank = None
        while self.rank is None:
            # Incase we want to slow down the player
            time.sleep(self.delay)
            # Acquire the lock for moskaGame, returns true if the lock was acquired, and False if there was a problem
            with self.moskaGame.get_lock(self) as ml:
                if self.moskaGame.EXIT_FLAG:
                    self.EXIT_STATUS = 2
                    break
                # Keep track of target changes.
                target = self.moskaGame.get_target_player()
                if target is not curr_target:
                    turns_taken_for_this_player = 0
                    curr_target = target
                turns_taken_for_this_player += 1
                
                if not ml:
                    continue

                msgd = {
                    "target" : curr_target.name,
                    "cards_to_fall" : self.moskaGame.cards_to_fall,
                    "fell_cards" : self.moskaGame.fell_cards,
                    "hand" : self.hand,
                    "Deck" : len(self.moskaGame.deck),
                    }
                # If a human is playing, then we print the values to terminal
                if self.requires_graphic:
                    print(f"{self.name} playing...",flush=True)
                    print(self.moskaGame,flush=True)
                    # Only print the players personal information, if the name has 'Human'
                    if "Human" in self.name:
                        print(msgd, flush=True)
                    self.moskaGame.glog.debug(f"{self.name} playing...")
                    self.moskaGame.glog.debug(self.moskaGame)
                    self.moskaGame.glog.debug(msgd)
                # If there is only 1 active player in the game, the player is last
                if len(self.moskaGame.get_players_condition(lambda x : x.rank is None)) <= 1:
                    self._set_rank()
                    break
                self.plog.debug(f"{msgd}")
                try:
                    # Try to play moves, as long as a valid move is played.
                    # At _play_move, the self.ready is set to True
                    success, msg = self._play_move()    # Return (True, "") if a valid move, else (False, "<error>")
                    # NOTE: If a deterministic player can make invalid moves, it will get stuck in this loop.
                    # Players should only make valid moves, because currently invalid moves are handled by errors,
                    # and make the game slower when triggered.
                    while not success:
                        self.plog.warning(msg)
                        self.ready = False
                        self.plog.debug(f"Player set to not ready, because of: {msg}")
                        print(msg, flush=True)
                        success, msg = self._play_move()
                # If an exception was raised for some reason. Invalid move errors are caught, and do not end up here.
                except Exception as msg:
                    self.plog.error(traceback.format_exc())
                    self.plog.error(msg)
                    self.EXIT_STATUS = 2
                    break
                # The target player is not ready, until they play "EndTurn"
                # Value of self.min_turns doesn't seem to have an effect.
                if turns_taken_for_this_player < self.min_turns or (self is curr_target and self.moskaGame.cards_to_fall):
                    self.ready = False
                    self.plog.debug(f"Player set to NOT ready, because {f'{turns_taken_for_this_player} < {self.min_turns}' if turns_taken_for_this_player < self.min_turns else 'cards_to_fall'}")
                # Set the players rank
                self._set_rank()
                # Check if self has finished, and hasn't played "EndTurn"
                # 'EndTurn' was last played, if the target changed during _play_move
                if self.rank is not None and self is self.moskaGame.get_target_player():
                    success, msg = self.moskaGame._make_move("EndTurn",[self,[]])
                    if not success:
                        self.plog.error(msg)
                        self.EXIT_STATUS = 2
                        break
                    self.ready = True
        if self.EXIT_STATUS == 1:
            self.plog.info(f"Finished as {self.rank}")
            self.ready = True
        elif self.EXIT_STATUS == 2:
            self.plog.info("Finished with error")
        return
    
    def _check_can_fall_card(self, played_card : Card, fall_card : Card) -> bool:
        """ Returns true, if the played_card, can fall the fall_card.
        The played card can fall fall_card, if:
        - The played card has the same suit and is greater than fall_card
        - If the played_card is trump suit, and the fall_card is not.

        Args:
            played_card (Card): The card played from hand
            fall_card (Card): The card on the table
            trump (str): The trump suit of the current game

        Returns:
            bool: True if played_card can fall fall_card, false otherwise
        """
        return utils.check_can_fall_card(played_card,fall_card,self.moskaGame.trump)
    
    def _map_to_list(self,card : Card) -> List[Card]:
        """ Return a list of cards, that the input card can fall from moskaGame.cards_to_fall
        """
        return _map_to_list(card,self.moskaGame.cards_to_fall,self.moskaGame.trump)
    
    def _map_each_to_list(self,from_ = None, to = None) -> Dict[Card,List[Card]]:
        """Map each card in hand, to a list of cards on the table, that can be fallen.
        Returns a dictionary

        Returns:
            _type_: _description_
        """
        # Make a dictionary of 'card-in-hand' : List[card-on-table] pairs, to know what which cards can be fallen with which cards
        if from_ is None:
            from_ = self.hand.cards
        if to is None:
            to = self.moskaGame.cards_to_fall
        return _map_each_to_list(from_,to,self.moskaGame.trump)
    
    def _make_cost_matrix(self, from_ = None, to = None, scoring : Callable = None, max_val : int = 100000) -> np.ndarray:
        """ Create a matrix, from from_ to to. The lists from_ and to have to contain Card -instances.
        The index i,j will correspond to the assignment of the i-th card in from_ to j-th card from to.
        The value at index will be max_val, if the assignment i-j is not valid, i.e. card i cant kill card j.
        If the assignment i-j is valid, then the score of the assignment will be calculated with the 'scoring : Callable' argument.

        Args:
            from_ (List[Card], optional): The list of cards, which are used to kill (for example the players hand). Defaults to the players hand.
            to (List[Card], optional): The list of cards, that are being killed. Defaults to the un-killed cards on the table.
            scoring (Callable, optional): The callable used to assign a score to a VALID assignment 
                                        The score should be low for good assignments and high for bad, if used to solve a minimum linear sum assignment problem.
                                        The signature of the scoring should be 'scoring(from[i] : Card, to[j] : Card) -> float'.
            max_val (int, optional): A large number (doesn't support inf :( ), that is used as a filler score for invalid assignments. Defaults to 100000.

        Raises:
            AttributeError: If no scoring is specified.

        Returns:
            np.ndarray: A numpy array, representing the assignment matrix (can be non-square).
        """
        if scoring is None:
            try:
                scoring = self._calc_assign_score
            except AttributeError as ae:
                raise AttributeError(f"{ae}\nSpecify 'scoring : Callable' as an argument or have a _calc_score -method in self.")
        if from_ is None:
            from_ = self.hand.cards
        if to is None:
            to = self.moskaGame.cards_to_fall
        return _make_cost_matrix(from_,to, self.moskaGame.trump,scoring,max_val)
    

    #### ABSTRACT METHODS ####
    
    @abstractmethod
    def choose_move(self,playable : List[str]) -> str:
        """ Select a move to play, such as "PlayToSelf", "PlayFallFromDeck", etc.

        Args:
            Playable (List[str]): A list of the available plays as string identifiers.

        Returns:
            str: The move identifier from 'playable' that you want to play.
        """
        pass
    
    @abstractmethod
    def end_turn(self) -> List[Card]:
        """Return which cards you want to pick from the table WHEN you have finished your turn as target.
        
        Returns:
            list : List of cards to pick from the table
        
        """
        pass
    
    @abstractmethod
    def play_fall_card_from_hand(self) -> Dict[Card,Card]:
        """Return a dictionary of card_in_hand : card_in_table -pairs, denoting which card is used to fall which card on the table.
        This function is called when the player has decided to play from their hand.

        Returns:
            Dict[Card,Card]: Card-in-hand - card-on-table pairs
        """
        pass
    
    @abstractmethod
    def deck_lift_fall_method(self, deck_card : Card) -> Tuple[Card,Card]:
        """A function to determine which card on the table will fall, if a random card from the deck is lifted.
        Function takes a card -instance as argument, and returns a pair (card_from_deck , card_on_table) in the same order,
        determining which card on the table to fall.
        
        This function is called, when the player decides to koplata AND the koplattu card can fall a card on the table.
        If the koplattu card can't fall any card on the table, then the card is just placed on the table, WITHOUT CALLING THIS FUNCTION.
        
        When this function is called, it is guaranteed, that there is a card on the table that can be falled with the card from deck.
        
        NOTE: If this function returns a pair, that doesn't work, then a random card is chosen.

        Args:
            deck_card (Card): The lifted card from the deck

        Returns:
            tuple(Card,Card): The input card from deck, the card on the table.
        """
        pass
    
    @abstractmethod
    def play_to_self(self) -> List[Card]:
        """Which cards from hand to play to table as a list of Card -instances.
        
        Returns:
            List[Card]: list of cards played to self
        """
        pass
    
    @abstractmethod
    def play_initial(self) -> List[Card]:
        """Return a list of cards that will be played to target on an initiating turn. AKA playing to an empty table.
        This function should always return a non empty list.

        Returns:
            List[Card]: _description_
        """
        pass
    
    @abstractmethod
    def play_to_target(self) -> List[Card]:
        """Return a list of cards, that will be played to target.
        This function can be called, when there are cards on the table, and you can play cards to a target

        Returns:
            List[Card]: List of Card -instances from hand, that can be played to the target.
        """
        pass
