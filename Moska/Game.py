from . import utils
from .BasePlayer import BasePlayer
from typing import Callable, List, TYPE_CHECKING
from .Deck import Card, StandardDeck
import threading

class MoskaGame:
    players : List[BasePlayer] = []
    triumph : str = ""
    triumph_card = None
    cards_to_fall = []
    fell_cards = []
    turnCycle = utils.TurnCycle([],ptr = 0)
    deck = None
    threads = []
    main_lock : threading.RLock = None
    def __init__(self,deck : StandardDeck):
        """Create a MoskaGame -instance.

        Args:
            deck (StandardDeck): The deck instance, from which to draw cards.
        """
        self.deck = deck
        
    def _create_locks(self) -> None:
        """ Initialize the RLock for the game. """
        self.main_lock = threading.RLock()
        return
        
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
        if not self.threads:
            self._init_player_threads()
        for thr in self.threads:
            thr.start()
        return
    
    def _init_player_threads(self) -> None:
        """ initialize the player threads. After this, the threads are ready to be run. """
        for pl in self.players:
            pl._start()
            self.threads.append(pl.thread)
    
    def _join_threads(self) -> None:
        """ Join all threads. """
        for trh in self.threads:
            trh.join()

    
    def add_player(self,player : BasePlayer) -> None:
        """Add a player to this MoskaGame -instance.
        Adds the player to the list of players, and to the TurnCycle -instance attribute

        Args:
            player (Player.MoskaPlayerBase): The player to add to the MoskaGame -instance.
        """
        self.players.append(player)
        self.turnCycle.add_to_population(player)
        return
    
    
    def get_initiating_player(self) -> BasePlayer:
        """ Return the player, whose turn it is/was to initiate the turn aka. play to an empty table. """
        active = self.get_target_player()
        ptr = int(self.turnCycle.ptr)
        out = self.turnCycle.get_prev_condition(cond = lambda x : x.rank is None and x is not active,incr_ptr=False)
        #self.turnCycle.set_pointer(ptr)    #TODO: This should work fine without this line.
        assert self.turnCycle.ptr == ptr
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
        self.triumph_card = triumph_card
        self.deck.place_to_bottom(self.triumph_card)
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
        print("Starting a game of Threaded Moska...")
        self._start_player_threads()
        #while len(self.get_players_condition(cond = lambda x : x.rank is None)) > 1:
        #    pass
        self._join_threads()
        print("Final Ranking: ")
        ranks = [(p.name, p.rank) for p in self.players]
        ranks = sorted(ranks,key = lambda x : x[1] if x[1] is not None else float("inf"))
        for p,rank in ranks:
            print(f"#{rank} - {p}")
        return ranks