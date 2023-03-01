import contextlib
import os
import sys
import time
from Moska.Game.GameState import FullGameState, GameState
from ..Player.MoskaBot3 import MoskaBot3
from . import utils
from ..Player.MoskaBot0 import MoskaBot0
from ..Player.AbstractPlayer import AbstractPlayer
from ..Player.MoskaBot1 import MoskaBot1
from ..Player.MoskaBot2 import MoskaBot2
from ..Player.RandomPlayer import RandomPlayer
from ..Player.HeuristicEvaluatorBot import HeuristicEvaluatorBot
from ..Player.NNEvaluatorBot import NNEvaluatorBot
from ..Player.NNHIFEvaluatorBot import NNHIFEvaluatorBot
import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from .Deck import Card, StandardDeck
from .CardMonitor import CardMonitor
import threading
import logging
import random
from contextlib import redirect_stdout
#import tensorflow as tf is done at set_model_vars_from_path IF a path is given.
# This is to gain a speedup when not using tensorflow
from .Turns import PlayFallFromDeck, PlayFallFromHand, PlayToOther, InitialPlay, EndTurn, PlayToSelf, Skip, PlayToSelfFromDeck


class MoskaGame:
    players : List[AbstractPlayer] = [] # List of players, with unique pids, and cards already in hand
    triumph : str = ""              # Triumph suit, set when the moskaGame is started
    triumph_card : Card = None             # Triumph card
    cards_to_fall : List[Card] = []              # Current cards on the table, for the target to fall
    fell_cards : List[Card] = []                 # Cards that have fell during the last turn
    turnCycle = utils.TurnCycle([],ptr = 0) # A TurnCycle instance, that rotates from the last to the first, created when players defined
    deck  : StandardDeck = None                             # The deck belonging to the moskaGame. 
    threads : Dict[int,AbstractPlayer] = {}
    log_file : str = ""
    log_level = logging.INFO
    name : str = __name__
    glog : logging.Logger = None
    main_lock : threading.RLock = None
    lock_holder = None
    turns : dict = {}
    timeout : float = 3
    random_seed = None
    nplayers : int = 0
    card_monitor : CardMonitor = None
    model_paths : List[str] = [""]
    __prev_lock_holder__ = None
    GATHER_DATA : bool = True
    EXIT_FLAG = False
    IS_RUNNING = False
    def __init__(self,
                 deck : StandardDeck = None,
                 players : List[AbstractPlayer] = [],
                 nplayers : int = 0,
                 log_file : str = "",
                 log_level = logging.INFO,
                 timeout=3,
                 random_seed=None,
                 gather_data : bool = True,
                 model_paths : List[str] = [""],
                 ):
        """Create a MoskaGame -instance.

        Args:
            deck (StandardDeck): The deck instance, from which to draw cards.
        """
        self.GATHER_DATA = gather_data
        self.IS_RUNNING = False
        #print(self.input_details)
        self.threads = {}
        if self.players or self.nplayers > 0:
            print("LEFTOVER PLAYERS FOUND!!!!!!!!!!!!!")
        if self.card_monitor is not None:
            print("LEFTOVER card_monitor!!!!!!!!!!!!1")
        if self.threads:
            print("LEFTOVER THREAD!!!!!")
        self.log_level = log_level
        self.log_file = log_file if log_file else os.devnull
        self.model_paths = model_paths
        self.set_model_vars_from_paths()
        self.random_seed = random_seed if random_seed else int(100000*random.random())
        self.deck = deck if deck else StandardDeck(seed = self.random_seed)
        self.players = players if players else self._get_random_players(nplayers)
        self.timeout = timeout
        self.EXIT_FLAG = False
        self.card_monitor = CardMonitor(self)
        self._set_turns()
        #self.model = tf.keras.models.load_model("/home/ilmari/python/moska/Model5-300/model.h5")

    def set_model_vars_from_paths(self):
        if isinstance(self.model_paths,str):
            self.model_paths = [self.model_paths] if self.model_paths else []
        # remove "" paths
        self.model_paths = [path for path in self.model_paths if path]
        self.interpreters = []
        self.input_details = []
        self.output_details = []
        if not self.model_paths:
            self.glog.info("No model paths given, not loading any models.")
            return
        self.glog.debug("Importing Tensorflow and loading models from paths: {}".format(self.model_paths))
        # NOTE: Import tensorflow only if there are models to load! This speeds up the process.
        # This also allows the user to run the game without tensorflow installed.
        # Furthermore, Tensorflow cannot be run with optimizations (-OO flag),
        # So this allows us to simulate games without tensorflow bots with optmizations
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        for path in self.model_paths:    
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            self.interpreters.append(interpreter)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            self.output_details.append(output_details)
            self.input_details.append(input_details)
        self.glog.info(f"Loaded {self.model_paths} models.")
        self.glog.debug(f"Input details: {self.input_details}")
        self.glog.debug(f"Output details: {self.output_details}")
        return
    
    def model_predict(self, X : np.ndarray, model_id : (str or int) = "all"):
        self.threads[threading.get_native_id()].plog.debug(f"Predicting with model, X.shape = {X.shape}")
        if model_id == "all":
            model_id = list(range(len(self.interpreters)))
        if isinstance(model_id,int):
            model_id = [model_id]
        if isinstance(model_id,str):
            try:
                model_id = [self.model_paths.index(model_id)]
            except:
                raise Exception(f"Could not find model path {model_id} in {self.model_paths}")
        output_data = []
        if not isinstance(X,np.ndarray):
            try:
                self.glog.debug(f"Converting X {type(X)} to np.ndarray")
                X = np.array(X)
            except:
                raise Exception(f"Could not convert {X} to np.ndarray")
        if not model_id:
            raise Exception(f"model_id is empty: {model_id}")
        if not X.shape:
            raise Exception(f"X.shape is empty: {X.shape}")
        if not self.interpreters:
            raise Exception("No model found for prediction. Model paths: {}".format(self.model_paths))
        for m_id,model_info in enumerate(zip(self.interpreters,self.input_details,self.output_details)):
            interpreter,input_details,output_details = model_info
            if m_id not in model_id:
                continue
            if len(X.shape) != len(input_details[0]["shape"]):
                self.glog.info(f"Expanding X.shape from {X.shape} to {input_details[0]['shape']}")
                X = np.expand_dims(X, axis=-1)
                self.glog.info(f"Expanded X.shape to {X.shape}")
            interpreter.resize_tensor_input(input_details[0]["index"],X.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], X)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            output_data.append(out)
        return np.mean(output_data,axis=0)
    
    def _set_turns(self):
        self.turns = {
        "PlayFallFromHand" : PlayFallFromHand(self),
        "PlayFallFromDeck" : PlayFallFromDeck(self),
        "PlayToOther" : PlayToOther(self),
        "PlayToSelf" : PlayToSelf(self),
        "InitialPlay" : InitialPlay(self),
        "EndTurn": EndTurn(self),
        "Skip":Skip(self),
        "PlayToSelfFromDeck":PlayToSelfFromDeck(self)
        }
        return


    def __getattribute____(self, __name: str) -> Any:
        """ This is a fail safe to prevent dangerous access to MoskaGame
        attributes from outside the main or player threads, when the game is locked.
        """
        non_accessable_attributes = ["card_monitor", "deck", "turnCycle", "cards_to_fall", "fell_cards", "players"]
        if __name in non_accessable_attributes and self.threads and threading.get_native_id() != self.lock_holder:
            raise threading.ThreadError(f"Getting MoskaGame attribute with out lock!")
        return object.__getattribute__(self,__name)
    
    def __setattr__(self, name, value):
        """ Prevent access to MoskaGame attributes from outside the main or player threads, when the game is locked.
        Also used for ensuring that inter-related attributes are set correctly.
        """
        if name == "EXIT_FLAG":
            super.__setattr__(self, name, value)
            return
        if name != "lock_holder" and self.threads and threading.get_native_id() != self.lock_holder:
            raise threading.ThreadError(f"Setting MoskaGame attribute with out lock!")
        super.__setattr__(self, name, value)
        # If setting the players, set the turnCycle and the new deck
        if name == "players":
            self._set_players(value)
            self.glog.debug(f"Set players to: {value}")
        # If setting the log_file, set the logger
        if name == "log_file" and value:
            assert isinstance(value, str), f"'{name}' of MoskaGame attribute must be a string"
            self.name = value.split(".")[0]
            self._set_glogger(value)
            self.glog.debug(f"Set GameLogger (glog) to file {value}")
        # If setting nplayers, create random players and set self.players
        if name == "nplayers":
            self.players = self.players if self.players else self._get_random_players(value)
            self.glog.debug(f"Created {value} random players.")
        # If setting the random seed, set the random seed
        if name == "random_seed":
            random.seed(value)     
            self.glog.info(f"Set random_seed to {self.random_seed}")
        return
    
    def _set_players(self,players : List[AbstractPlayer]) -> None:
        """Here self.players is already set to players
        """
        assert isinstance(players, list), f"'players' of MoskaGame attribute must be a list"
        self.deck = StandardDeck(seed=self.random_seed)
        for pl in players:
            pl.moskaGame = self
        self.turnCycle = utils.TurnCycle(players)
        return
        
    @classmethod
    def _get_random_players(cls,n, player_types : List[Callable] = [],**plkwargs) -> List[AbstractPlayer]:
        """ Get a list of AbstractPlayer  subclasses.
        The players will be dealt cards from a new_deck if called from an instance.

        Args:
            n (int): Number of players to create
            player_types (list[Callable], optional): The player types to use. Defaults to all.

        Returns:
            _type_: _description_
        """
        players = []
        if not player_types:
            player_types = [MoskaBot0,MoskaBot1, MoskaBot2, MoskaBot3, RandomPlayer, HeuristicEvaluatorBot, NNEvaluatorBot, NNHIFEvaluatorBot]
        for i in range(n):
            rand_int = random.randint(0, len(player_types)-1)
            player = player_types[rand_int]()
            players.append(player)
        return players
    
    def _set_glogger(self,log_file : str) -> None:
        """Set the games logger `glog`.

        Args:
            log_file (str): Where to write the games log
        """
        self.glog = logging.getLogger(self.name)
        self.glog.setLevel(self.log_level)
        fh = logging.FileHandler(log_file,mode="w",encoding="utf-8")
        formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
        fh.setFormatter(formatter)
        self.glog.addHandler(fh)
        return
    
    def _create_locks(self) -> None:
        """ Initialize the RLock for the game. """
        self.main_lock = threading.RLock()
        self.glog.debug("Created RLock")
        return
    
    @contextlib.contextmanager
    def get_lock(self,player=None):
        """A wrapper around getting the moskagames main_lock.
        Sets the lock_holder to the obtaining threads id

        Args:
            player (_type_): _description_

        Yields:
            _type_: _description_
        """
        with self.main_lock as lock:
            self.lock_holder = threading.get_native_id()
            if self.lock_holder == self.__prev_lock_holder__:
                self.lock_holder = None
                yield False
                return
            og_state = len(self.cards_to_fall + self.fell_cards)
            if self.lock_holder not in self.threads:
                self.lock_holder = None
                print(f"Game {self.log_file}: Couldn't find lock holder id {self.lock_holder}!")
                yield False
                return
            if self.EXIT_FLAG:
                self.lock_holder = None
                yield False
                return
            # Here we tell the player that they have the key
            yield True
            state = len(self.cards_to_fall + self.fell_cards)
            if og_state != state:
                self.glog.info(f"{self.threads[self.lock_holder].name}: new board: {self.cards_to_fall}")
            assert len(set(self.cards_to_fall)) == len(self.cards_to_fall), f"Game log {self.log_file} failed, DUPLICATE CARD"
            pl = self.threads[self.lock_holder]
            if False and isinstance(pl, AbstractPlayer):
                inp = pl.state_vectors[-1]
                #print(inp)
                possib_to_not_lose = self.model.predict(np.array([inp]),verbose=0)
                print(f"{pl.name} has {possib_to_not_lose[0]} chance of not losing")
            self.__prev_lock_holder__ = self.lock_holder
            self.lock_holder = None
        return
    
    def _make_move(self,move,args) -> Tuple[bool,str]:
        """ This is called from a AbstractPlayer -instance
        """
        if self.lock_holder != threading.get_native_id():
            raise threading.ThreadError(f"Making moves is supposed to be implicit and called in a context manager after acquiring the games lock")
        if move not in self.turns.keys():
            raise NameError(f"Attempted to make move '{move}' which is not recognized as a move in Turns.py")
        move_call = self.turns[move]
        self.glog.debug(f"Player {self.threads[self.lock_holder].name} called {move} with args {args}")
        try:
            move_call(*args)  # Calls a class from Turns, which raises AssertionError if the move is not playable
        except AssertionError as ae:
            self.glog.warning(f"{self.threads[threading.get_native_id()].name}:{ae}")
            return False, str(ae)
        except TypeError as te:
            self.glog.warning(f"{self.threads[threading.get_native_id()].name}:{te}")
            return False, str(te)
        self.card_monitor.update_from_move(move,args)
        return True, ""
    
    def _make_mock_move(self,move,args,state_fmt="FullGameState") -> FullGameState:
        """ Makes a move, like '_make_move', but returns the game state after the move and restores self and attributes to its original state.
        """
        state = FullGameState.from_game(self)
        #self.glog.debug("Saved state data. Setting logger to level WARNING for Mock move")
        player = args[0]
        game_log_level = self.glog.getEffectiveLevel()
        player_log_level = player.plog.getEffectiveLevel()
        # Disable logging for Mock move
        self.glog.setLevel(logging.WARNING)
        player.plog.setLevel(logging.WARNING)

        # Normally play the move; change games state precisely as it would actually change.
        success, msg = self._make_move(move,args)
        self.glog.setLevel(game_log_level)
        player.plog.setLevel(player_log_level)
        if not success:
            raise AssertionError(f"Mock move failed: {msg}")
        if state_fmt == "FullGameState" or "FullGameState-Depr":
            new_state = FullGameState.from_game(self,copy=True)
        elif state_fmt == "GameState":
            new_state = GameState.from_game(self)
        else:
            raise NameError(f"Argument 'state_fmt' was not recognized. Given argument: {state_fmt}")
        # Save the new game state, for evaluation of the move
        state.restore_game_state(self,check=False)
        is_eq, msg = state.is_game_equal(self,return_msg=True)
        if not is_eq:
            raise AssertionError(f"Mock move failed: {msg}")
        # Return the new_state
        return new_state
        
        
    
    
    def __repr__(self) -> str:
        """ What to print when calling print(self) """
        s = f"Triumph card: {self.triumph_card}\n"
        s += f"Deck left: {len(self.deck.cards)}\n"
        if True and all((pl.requires_graphic for pl in self.players)) and self.model_paths:
            player_evals = []
            state = FullGameState.from_game(self,copy=False)
            for pl in self.players:
                # If player doesn't have model_id or pred_format, use default values
                if not hasattr(pl,"model_id") or not hasattr(pl,"pred_format"):
                    pl_to_copy = self.get_players_condition(lambda x : x.name != pl.name and hasattr(x,"model_id") and hasattr(x,"pred_format"))[0]
                    model_id = pl_to_copy.model_id
                    pred_format = pl_to_copy.pred_format
                else:
                    model_id = pl.model_id
                    pred_format = pl.pred_format
                evaluation = self.model_predict(np.array(state.as_perspective_vector(pl,fmt=pred_format),dtype=np.float32),model_id=model_id)
                player_evals.append(round(float(evaluation),2))
        for pid,pl in enumerate(self.players):
            s += f"{pl.name}{'*' if pl is self.get_target_player() else ''}({player_evals[pid]}) : {self.card_monitor.player_cards[pl.name]}\n"
        s += f"Cards to fall : {self.cards_to_fall}\n"
        s += f"Fell cards : {self.fell_cards}\n"
        return s
    
    def _start_player_threads(self) -> None:
        """ Starts all player threads. """
        self.cards_to_fall.clear()
        self.fell_cards.clear()
        # Add self to allowed threads
        self.threads[threading.get_native_id()] = self
        with self.get_lock() as ml:
            for pl in self.players:
                tid = pl._start()
                self.threads[tid] = pl
            self.glog.debug("Started player threads")
            assert len(set([pl.pid for pl in self.players])) == len(self.players), f"A non-unique player id ('pid' attribute) found."
            self.card_monitor.start()
            self.IS_RUNNING = True
        return
    
    def _join_threads(self) -> None:
        """ Join all threads. """
        start = time.time()
        while time.time() - start < self.timeout and any([pl.thread.is_alive() for pl in self.players]):
            time.sleep(0.1)
            if any((pl.EXIT_STATUS == 2 for pl in self.players)):
                with self.get_lock() as ml:
                    print(f"Game with log {self.log_file} failed.")
                    self.glog.error(f"Game FAILED. Exiting.")
                    self.EXIT_FLAG = True
                return False
        if any((pl.EXIT_STATUS != 1 for pl in self.players)):
            #self.lock_holder = threading.get_native_id()
            print(f"Game with log {self.log_file} timedout.")
            self.glog.error(f"Game timedout after {self.timeout} seconds. Exiting.")
            self.EXIT_FLAG = True
            return False
        self.glog.info("Threads finished")
        return True
    
    
    def get_initiating_player(self) -> AbstractPlayer:
        """ Return the player, whose turn it is/was to initiate the turn aka. play to an empty table. """
        active = self.get_target_player()
        ptr = int(self.turnCycle.ptr)
        out = self.turnCycle.get_prev_condition(cond = lambda x : x.rank is None and x is not active,incr_ptr=False)
        #self.turnCycle.set_pointer(ptr)    #TODO: This should work fine without this line.
        assert self.turnCycle.ptr == ptr, "Problem with turnCycle"
        return out
    
    def get_players_condition(self, cond : Callable = lambda x : True) -> List[AbstractPlayer]:
        """ Get a list of players that return True when condition is applied.

        Args:
            cond (Callable): The condition to be applied to each AbstractPlayer -instance. Defaults to lambda x : True.

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
        
    def get_target_player(self) -> AbstractPlayer:
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
        self._orig_triumph_card = triumph_card
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
    
    def get_player_state_vectors(self, shuffle = True, balance = True) -> List[List]:
        losers = []
        not_losers = []
        # Combine the players vectors into one list
        for pl in self.players:
            # If the player lost, append 0 to the end of the vector, else append 1
            pl_not_lost = 0 if pl.rank == len(self.players) else 1
            for state in pl.state_vectors:
                if pl_not_lost == 0:
                    losers.append(state + [0])
                else:
                    not_losers.append(state + [1])
        if balance:
            state_results = losers + random.sample(not_losers,min(len(losers),len(not_losers)))
        if shuffle:
            random.shuffle(state_results)
        return state_results
        
        
    def get_random_file_name(self,min_val = 1, max_val = 1000000000):
        fname = "data_"+str(random.randint(int(min_val),int(max_val)))+".out"
        tries = 0
        while os.path.exists(fname) and tries < 1000:
            fname = "data_"+str(random.randint(0,max_val))+".out"
            tries += 1
        if tries == 1000:
            print("Could not find a unique file name. Saving to custom file name.")
            fname = "data_new_"+str(random.randint(0,max_val))+".out"
        return fname
    
    def start(self) -> bool:
        """The main method of MoskaGame. Sets the triumph card, locks the game to avoid race conditions between players,
        initializes and starts the player threads.
        After that, the players play the game, only one modifying the state of the game at a time.

        Returns:
            True when finished
        """
        self._set_triumph()
        self._create_locks()
        self.glog.info(f"Starting the game with seed {self.random_seed}...")
        self._start_player_threads()
        self.glog.info(f"Started moska game with players {[pl.name for pl in self.players]}")
        # Wait for the threads to finish
        success = self._join_threads()
        if not success:
            #del self
            #self.IS_RUNNING = False
            return None
        #cards_in_card_monitor = list(self.card_monitor.cards_fall_dict.keys())
        #cards_left = self.get_players_condition(lambda x : x.rank == len(self.players))[0].hand.cards + self.fell_cards + self.cards_to_fall
        self.glog.info("Final ranking: ")
        ranks = [(p.name, p.rank) for p in self.players]
        state_results = []
        if self.GATHER_DATA:
            state_results = self.get_player_state_vectors()
            with open("Vectors/"+self.get_random_file_name(),"w") as f:
                data = str(state_results).replace("], [","\n")
                data = data.strip("[]")
                f.write(data)
        
        ranks = sorted(ranks,key = lambda x : x[1] if x[1] is not None else float("inf"))
        for p,rank in ranks:
            self.glog.info(f"#{rank} - {p}")
        return ranks