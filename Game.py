import queue
import utils
import Player
from typing import List
import threading

class MoskaGame:
    players : List[Player.MoskaPlayerBase] = []
    triumph : str = ""
    triumph_card = None
    cards_to_fall = []
    fell_cards = []
    turnCycle = utils.TurnCycle([],ptr = 0)
    deck = None
    def __init__(self,deck):
        self.deck = deck
    
    def add_player(self,player : Player.MoskaPlayerBase):
        """ Add a player to the list and to the turncycle"""
        self.players.append(player)
        self.turnCycle.add_to_population(player)
    
    def __repr__(self):
        """ Print the status of the game"""
        s = f"Triumph card: {self.triumph_card}\n"
        for pl in self.players:
            s += f"{pl.name}{'*' if pl is self.get_active_player() else ''} : {pl.hand}\n"
        s += f"Cards to fall : {self.cards_to_fall}\n"
        s += f"Fell cards : {self.fell_cards}\n"
        return s
    
    def get_initiating_player(self):
        active = self.get_active_player()
        ptr = int(self.turnCycle.ptr)
        out = self.turnCycle.get_prev_condition(cond = lambda x : x.rank is None and x is not active,incr_ptr=False)
        self.turnCycle.set_pointer(ptr)
        assert self.turnCycle.ptr == ptr
        return out
    
    def get_players_condition(self,cond = lambda x : True):
        """ get players that match the condition """
        return list(filter(cond,self.players))
    
    def add_cards_to_fall(self,add : list):
        self.cards_to_fall += add
        
    def get_active_player(self):
        return self.turnCycle.get_at_index()
        #return self.get_players(cond = lambda x : x.pid == self.moskaTable.turn_pid)[0]     # Get the player to whom the cards are played to
    
    def set_triumph(self):
        assert len(self.players) > 1 and len(self.players) < 8, "Too few or too many players"
        self.triumph_card = self.deck.pop_cards(1)[0]
        self.triumph = self.triumph_card.suit
        self.deck.place_to_bottom(self.triumph_card)
    
    def initial_play(self):
        prev_player = self.get_initiating_player()
        if not prev_player:
            return
        prev_player._play_initial()
        print(self)
    
    def let_others_play(self):
        player = self.get_active_player()
        status = self.cards_to_fall.copy()
        lcount = 0
        while lcount < 1 or not all([og == n for og,n in zip(status, self.cards_to_fall)]):
            for other_player in self.players:
                if other_player is not player and other_player.rank is None:
                    other_player._play_to_target()
                    print(f"Player {other_player.name} playing to {player.name}")
                    print(self)
            status = self.cards_to_fall.copy()
            lcount += 1
    
    def active_player_fall_cards(self):
        print("Falling cards...")
        player = self.get_active_player()
        player._play_fall_card_from_hand()
        print(self)
    
    """
    def start(self):
        self.set_triumph()
        print("Started a game of Moska")
        print(self)
        turn_count = 0
        while len(self.get_players_condition(cond = lambda x : x.rank is None)) > 1:
            print(f"Cards left in deck: {len(self.deck)}")
            print(f"Player {self.get_initiating_player().name} ", end="")
            print(f"playing to {self.get_active_player().name}")
            assert not self.get_active_player().rank is not None, "The player is no longer in the game"
            self.initial_play()
            while True:
                self.let_others_play()
                status = self.cards_to_fall.copy()
                self.active_player_fall_cards()
                # If no cards were played
                if all([og == n for og,n in zip(status, self.cards_to_fall)]):
                    break
            active_player = self.get_active_player()
            lifted = active_player._end_turn()
            print(f"********* Player {active_player.pid} finished turn.*********\n\n")
            turn_count += 1
            self.turnCycle.get_next_condition(cond = lambda x : x.rank is None and x is not active_player)
            if lifted:
                self.turnCycle.get_next_condition(cond = lambda x : x.rank is None)
        print("Game finished with ranks:\n")
        ranks = {pid : rank for pid,rank in zip((pl.pid for pl in self.players),(pl.rank for pl in self.players))}
        for pid,rank in ranks.items():
            print(f"P{pid} : {rank}")
            
        print(f"Total amount of turns (lifting or falling cards): {turn_count}")
    """   
class MoskaGameThreaded(MoskaGame):
    threads = []
    main_lock : threading.RLock = None
    
    def _create_locks(self):
        self.main_lock = threading.RLock()
    
    def init_player_threads(self):
        for pl in self.players:
            pl.start()
            self.threads.append(pl.thread)
    
    def join_threads(self):
        for trh in self.threads:
            trh.join()
    
    def start_player_threads(self):
        for thr in self.threads:
            thr.start()
    
    def start(self):
        self.set_triumph()
        self._create_locks()
        print("Starting a game of Threaded Moska...")
        self.init_player_threads()
        self.start_player_threads()
        assert self.check_players_threaded(), "Some of the players in the game are not of type MoskaPlayerThreadedBase."
        while len(self.get_players_condition(cond = lambda x : x.rank is None)) > 1:
            pass
        self.join_threads()
        print("Final Ranking: ")
        ranks = [(p.name, p.rank) for p in self.players]
        ranks = sorted(ranks,key = lambda x : x[1] if x[1] is not None else float("inf"))
        for p,rank in ranks:
            print(f"#{rank} - {p}")
        exit()
        
     
    def check_players_threaded(self):
        return all((isinstance(pl,Player.MoskaPlayerThreadedBase) for pl in self.players))
        
        