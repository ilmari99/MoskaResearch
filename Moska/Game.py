import contextlib
from . import utils
from .Player.BasePlayer import BasePlayer
from .Player.MoskaBot1 import MoskaBot1
from typing import Any, Callable, Dict, List, TYPE_CHECKING
from .Deck import Card, StandardDeck
import threading
import logging
import random


class MoskaGame:
    players : List[BasePlayer] = [] # List of players, with unique pids, and cards already in hand
    triumph : str = ""              # Triumph suit, set when the moskaGame is started
    triumph_card : Card = None             # Triumph card
    cards_to_fall : List[Card] = []              # Current cards on the table, for the target to fall
    fell_cards : List[Card] = []                 # Cards that have fell during the last turn
    turnCycle = utils.TurnCycle([],ptr = 0) # A TurnCycle instance, that rotates from the last to the first, created when players defined
    deck  : StandardDeck = None                             # The deck belonging to the moskaGame. 
    threads : Dict[int,BasePlayer] = {}
    log_file : str = "gamelog.log"
    log_level = logging.INFO
    name : str = __name__
    glog : logging.Logger = None
    main_lock : threading.RLock = None
    lock_holder = None
    def __init__(self,
                 deck : StandardDeck = None,
                 players : List[BasePlayer] = [],
                 nplayers : int = 0,
                 log_file : str = "",
                 log_level = logging.INFO,):
        """Create a MoskaGame -instance.

        Args:
            deck (StandardDeck): The deck instance, from which to draw cards.
        """
        self.log_level = log_level
        self.log_file = log_file if log_file else self.log_file
        self.deck = deck if deck else StandardDeck()
        self.players = players if players else self._get_random_players(nplayers)
    
    def __setattr__(self, name, value):
        super.__setattr__(self, name, value)
        if name == "players":
            self._set_players(value)
            self.glog.debug(f"Set players to: {value}")
        if name == "log_file" and value:
            assert isinstance(value, str), f"'{name}' of MoskaGame attribute must be a string"
            self._set_glogger(value)
            self.glog.debug(f"Set GameLogger (glog) to file {value}")
        return
    
    def _set_players(self,players : List[BasePlayer]):
        """Here self.players is already set to players
        """
        assert isinstance(players, list), f"'players' of MoskaGame attribute must be a list"
        assert len(set([pl.pid for pl in self.players])) == len(self.players), f"A non-unique player id ('pid' attribute) found."
        self.deck = StandardDeck()
        for pl in players:
            pl.moskaGame = self
        self.turnCycle = utils.TurnCycle(players)
        
    @classmethod
    def _get_random_players(cls,n, player_types = []):
        """ Get a list of BasePlayer instances (or subclasses).
        The players will be dealt cards from 

        Args:
            n (_type_): _description_
            player_types (list, optional): _description_. Defaults to [].

        Returns:
            _type_: _description_
        """
        players = []
        if not player_types:
            player_types = [BasePlayer,MoskaBot1]
        for i in range(n):
            rand_int = random.randint(0, len(player_types)-1)
            player = player_types[rand_int](pid=i,debug=True)
            players.append(player)
        return players
    
    def _set_glogger(self,log_file):
        self.glog = logging.getLogger(self.name)
        self.glog.setLevel(self.log_level)
        fh = logging.FileHandler(log_file,mode="w",encoding="utf-8")
        formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
        fh.setFormatter(formatter)
        self.glog.addHandler(fh)
    
    def _create_locks(self) -> None:
        """ Initialize the RLock for the game. """
        self.main_lock = threading.RLock()
        self.glog.debug("Created RLock")
        return
    
    @contextlib.contextmanager
    def get_lock(self,player):
        """A wrapper around getting the moskagames main_lock

        Args:
            player (_type_): _description_

        Yields:
            _type_: _description_
        """
        with self.main_lock as lock:
            og_state = len(self.cards_to_fall + self.fell_cards)
            
            # Here we tell the player that they have the key
            self.lock_holder = threading.get_ident()
            yield lock
            self.lock_holder = None
            
            state = len(self.cards_to_fall + self.fell_cards)
            if og_state != state:
                self.glog.info(f"{player.name}: new board: {self.cards_to_fall}")
        return
    
    def _make_move(self,move : Callable) -> bool:
        """ This is called from a BasePlayer -instance """
        if self.lock_holder != threading.get_ident():
            raise threading.ThreadError(f"Making moves is supposed to be implicit and called in a context manager after acquiring the games lock")
        # TODO: check whether move is actually a move
        if move is None:
            move = int  # TODO: make this an actual move, now int is just a placeholder for doing nothing
        try:
            move()  # Calls a class from Turns, which raises AssertionError if the move is not playable
        except AssertionError as ae:
            self.glog.warning(f"{self.threads[threading.get_ident()].name}:{ae}")
            return False, str(ae)
        return True, ""
    
    
    def __repr__(self) -> str:
        """ What to print when calling print(self) """
        s = f"Triumph card: {self.triumph_card}\n"
        for pl in self.players:
            s += f"{pl.name}{'*' if pl is self.get_target_player() else ''} : {pl.hand}\n"
        s += f"Cards to fall : {self.cards_to_fall}\n"
        s += f"Fell cards : {self.fell_cards}\n"
        return s
    
    def _start_player_threads(self) -> None:
        """ Starts all player threads. """
        for pl in self.players:
            tid = pl._start()
            self.threads[tid] = pl
        self.glog.debug("Started player threads")
        return
    
    def _join_threads(self) -> None:
        """ Join all threads. """
        for pl in self.players:
            pl.thread.join()
        self.glog.debug("All threads finished")
        return
    
    
    def get_initiating_player(self) -> BasePlayer:
        """ Return the player, whose turn it is/was to initiate the turn aka. play to an empty table. """
        active = self.get_target_player()
        ptr = int(self.turnCycle.ptr)
        out = self.turnCycle.get_prev_condition(cond = lambda x : x.rank is None and x is not active,incr_ptr=False)
        #self.turnCycle.set_pointer(ptr)    #TODO: This should work fine without this line.
        assert self.turnCycle.ptr == ptr, "Problem with turnCycle"
        return out
    
    def get_players_condition(self, cond : Callable = lambda x : True) -> List[BasePlayer]:
        """ Get a list of players that return True when condition is applied.

        Args:
            cond (Callable): The condition to be applied to each BasePlayer -instance. Defaults to lambda x : True.

        Returns:
            List of players who satisfy condition.
        """
        return list(filter(cond,self.players))
    
    def add_cards_to_fall(self,add : List[Card]) -> None:
        """Add a list of cards to fall.

        Args:
            add (List): The list of cards to add to the table.
        """
        self.cards_to_fall += add
        return
        
    def get_target_player(self) -> BasePlayer:
        """Return the player, who is currently the target; To who cards are played to.

        Returns:
            Player.MoskaPlayerBase: the target player
        """
        return self.turnCycle.get_at_index()
    
    def _set_triumph(self) -> None:
        """Sets the triumph card of the MoskaGame.
        Takes the top-most card, checks the suit, and checks if any player has the 2 of that suit in hand.
        If some player has, then it swaps the cards.
        Places the card at the bottom of the deck.
        """
        assert len(self.players) > 1 and len(self.players) < 8, "Too few or too many players"
        triumph_card = self.deck.pop_cards(1)[0]
        self.triumph = triumph_card.suit
        p_with_2 = self.get_players_condition(cond = lambda x : any((x.suit == self.triumph and x.value==2 for x in x.hand.cards)))
        if p_with_2:
            assert len(p_with_2) == 1, "Multiple people have valtti 2 in hand."
            p_with_2 = p_with_2[0]
            p_with_2.hand.add([triumph_card])
            triumph_card = p_with_2.hand.pop_cards(cond = lambda x : x.suit == self.triumph and x.value == 2)[0]
            self.glog.info(f"Removed {triumph_card} from {p_with_2.name}")
        self.triumph_card = triumph_card
        self.deck.place_to_bottom(self.triumph_card)
        self.glog.info(f"Placed {self.triumph_card} to bottom of deck.")
        return
    
    def start(self) -> bool:
        """The main method of MoskaGame. Sets the triumph card, locks the game to avoid race conditions between players,
        initializes and starts the player threads.
        After that, the players play the game, only one modifying the state of the game at a time.

        Returns:
            True when finished
        """
        self._set_triumph()
        self._create_locks()
        self.glog.info(f"Starting the game...")
        self._start_player_threads()
        self._join_threads()
        self.glog.info("Final ranking: ")
        ranks = [(p.name, p.rank) for p in self.players]
        ranks = sorted(ranks,key = lambda x : x[1] if x[1] is not None else float("inf"))
        for p,rank in ranks:
            self.glog.info(f"#{rank} - {p}")
        return ranks