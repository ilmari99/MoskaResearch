import utils
import Player

class MoskaGame:
    players = []
    triumph = ""
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
            s += f"{pl.name}{'*' if self.turnCycle.ptr % len(self.players) == pl.pid else ''} : {pl.hand}\n"
        s += f"Cards to fall : {self.cards_to_fall}\n"
        s += f"Fell cards : {self.fell_cards}\n"
        return s
    
    def get_initiating_player(self):
        active = self.get_active_player()
        return self.turnCycle.get_prev_condition(cond = lambda x : x.rank is None and x is not active,incr_ptr=False)
    
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
class MoskaGame:
    players = []
    triumph = ""
    triumph_card = None
    cards_to_fall = []
    fell_cards = []
    turnCycle = utils.TurnCycle([],ptr = 0)
    deck = None
    current_turn = 0
    def __init__(self, deck):
        self.deck = deck
    
    def add_players(self,*players):
        for pl in players:
            self.players.append(pl)
    
    def __repr__(self):
        "" Print the status of the game""
        s = f"Triumph card: {self.triumph_card}\n"
        for pl in self.players:
            s += f"#{pl.pid}{'*' if self.turn_pid == pl.pid else ''} : {pl.hand}\n"
        s += f"Cards to fall : {self.cards_to_fall}\n"
        s += f"Fell cards : {self.fell_cards}\n"
        return s
    
    def get_players(self,cond = lambda x : True):
        return list(filter(cond,self.players))
    
    def get_active_players(self):
        return self.get_players(cond = lambda x : x.rank is None)
    
    def add_cards_to_fall(self,add):
        self.cards_to_fall += add
    
    def fall_card(self,played_card, fall_card):
        success = False
        # Jos kortit ovat samaa maata ja pelattu kortti on suurempi
        if played_card.suit == fall_card.suit and played_card.value > fall_card.value:
            success = True
        # Jos pelattu kortti on valttia, ja kaadettava kortti ei ole valttia
        elif played_card.suit == self.triumph and fall_card.suit != self.triumph:
                success = True
        return success
    
    def get_active_player(self):
        return self.get_players(cond = lambda x : x.pid == self.moskaTable.turn_pid)[0]     # Get the player to whom the cards are played to
    
    def next_turn_automatic(self,initial=False):
        print(self)
        turn_pid = self.turn_pid
        player = self.get_active_player()
        if player.rank is not None:
            self._end_turn()
            return
        # If no cards have been played to player in turn
        if initial:
            prev_player = player.get_prev_player()  # The previous player gets to make a free first move
            prev_player.play_to_another_automatic(initial=True,fits=len(player.hand))
            print(moskaTable)
        other_players = lambda : moskaTable.get_players(cond = lambda x : x.pid != turn_pid and x.rank is None)
        players_left = lambda : len(self.moskaTable.get_players(cond=lambda x : x.rank is None))
        played = True
        # The other players play cards to the target
        while played:
            played = False
            # If there is only one person with cards, he lost
            if players_left() <= 1:
                self.moskaTable.get_players(cond = lambda x : x.rank is None)[0].set_rank()
                break
            # If the table is full
            fits = len(player.hand) - len(self.moskaTable.cards_to_fall)
            if fits <= 0:
                break
            for ot_pl in other_players():
                res = ot_pl.play_to_another_automatic(fits=fits)
                played = played if played else res
            print(self.moskaTable)
        played = player.fell_cards_automatic()
        print(self.moskaTable)
        if played:
            self.next_turn_automatic()
        elif initial:
            self._end_turn()
            
        
    def _end_turn(self):
        player = self.get_active_player()
        lifted = False
        if self.moskaTable.cards_to_fall:
            lifted = True
            player.hand.add(self.moskaTable.cards_to_fall)
        self.moskaTable.cards_to_fall = []
        self.moskaTable.fell_cards = []
        if not lifted:
            self.moskaTable.turn_pid = player.get_next_player().pid
        else:
            self.moskaTable.turn_pid = player.get_next_player().get_next_player().pid
        player.hand.draw(6-len(player.hand))
"""