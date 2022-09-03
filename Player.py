from __future__ import annotations
import multiprocessing
from typing import TYPE_CHECKING
from copy import copy, deepcopy
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame fro typechekcing
    from Game import MoskaGame, MoskaGameThreaded
from Hand import MoskaHand
import utils
from Turns import PlayFallCardFromHand, PlayFallFromDeck, PlayToOther, InitialPlay, EndTurn
import threading
import time


class MoskaPlayerBase:
    """ The base class of a moska player. Subclassing this class, and writing new methods for:
    play_to_target() -> List of cards to play to a target
    play_initial() -> List of cards to play to a target, when you are initiating the turn
    deck_lift_fall_method(card_from_deck) -> (card_from_deck, to_which_card_on_table_you_want_to_play)
    play_fall_card_from_hand() -> Return a dictionary of values {card_from_hand : card_on_table}
    end_turn() -> List of cards that you want to pick from the table, when ending your turn.
    
    """
    hand = None
    pid = 0
    moskaGame = None
    rank = None
    thread = None
    name = ""
    ready = False
    def __init__(self,moskaGame : MoskaGame, pid : int = 0, name : str = ""):
        self.moskaGame = moskaGame
        self.hand = MoskaHand(moskaGame)
        self.pid = pid
        self.name = name if name else f"P{str(pid)}"
        self._playFallCardFromHand = PlayFallCardFromHand(self.moskaGame,self)
        self._playFallFromDeck = PlayFallFromDeck(self.moskaGame)
        self._playToOther = PlayToOther(self.moskaGame,self)
        self._initialPlay = InitialPlay(self.moskaGame,self)
        self._endTurn = EndTurn(self.moskaGame,self)
        
        
    def _set_pid(self,pid):
        self.pid = pid
    
    def _playable_values_to_table(self):
        return set([c.value for c in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards])
    
    def _playable_values_from_hand(self):
        """ Return a set of values, that can be played to target"""
        return self._playable_values_to_table().intersection([c.value for c in self.hand])
    
    def _fits_to_table(self):
        """ Return the number of cards playable to the active/target player"""
        target = self.moskaGame.get_active_player()
        return len(target.hand) - len(self.moskaGame.cards_to_fall)
    
    
    @utils.check_new_card
    def _play_to_target(self):
        """ This method is invoked to play the cards, chosen in 'play_to_target'"""
        play_cards = self.play_to_target()
        target = self.moskaGame.get_active_player()
        self._playToOther(target,play_cards)
    
    def play_to_target(self):
        """ Return a list of cards, that will be played to target. """
        playable_values = self._playable_values_from_hand()
        play_cards = []
        if playable_values:
            hand = self.hand.copy()
            play_cards = hand.pop_cards(cond=lambda x : x.value in playable_values,max_cards = self._fits_to_table())
        return play_cards
    
    @utils.check_new_card
    def _play_initial(self):
        target = self.moskaGame.get_active_player()
        play_cards = self.play_initial()
        self._initialPlay(target,play_cards)
    
    def play_initial(self):
        """ Return a list of cards that will be played to target on an initiating turn"""
        target = self.moskaGame.get_active_player()
        min_card = min([c.value for c in self.hand])
        hand = self.hand.copy()
        play_cards = hand.pop_cards(cond=lambda x : x.value == min_card,max_cards = self._fits_to_table())
        return play_cards
    
    def deck_lift_fall_method(self, deck_card):
        """ A function to determine which card will fall, if a random card from the deck is lifted.
        Function should take a card -instance as argument, and return a pair (card_from_deck , card_on_table) in the same order.
        Determining which card to fall."""
        for card in self.moskaGame.cards_to_fall:
            if utils.check_can_fall_card(deck_card,card):
                return (deck_card,card)
            
    @utils.check_new_card
    def _play_fall_from_deck(self):
        self._playFallFromDeck(fall_method=self.deck_lift_fall_method)

    @utils.check_new_card
    def _play_fall_card_from_hand(self):
        play_cards = self.play_fall_card_from_hand()
        self._playFallCardFromHand(play_cards)
    
    def play_fall_card_from_hand(self):
        """ Return a dictionary of card_in_hand : card_in_table -pairs, denoting which card is used to fall which card on the table."""
        play_cards = {}
        for fall_card in self.moskaGame.cards_to_fall:
            cards_that_fall = []
            for play_card in self.hand:
                success = utils.check_can_fall_card(play_card,fall_card,self.moskaGame.triumph)
                if success:
                    if play_card not in play_cards:
                        play_cards[play_card] = []
                    play_cards[play_card].append(fall_card)
        play_cards = {pc : min(fc) for pc,fc in play_cards.items()}
        pc_inv = {fc : pc for pc,fc in play_cards.items()}
        play_cards = {pc : fc for fc,pc in pc_inv.items()}
        return play_cards
    
    @utils.check_new_card
    def _end_turn(self):
        pick_cards = self.end_turn()
        self._endTurn(pick_cards)
        return bool(pick_cards)
    
    def end_turn(self):
        """ Return which cards you want to pick from the table after finishing yur turn."""
        pick_cards = self.moskaGame.cards_to_fall
        return pick_cards
        
    def _set_rank(self):
        if self.rank is None:   # if the player hasn't already finished
            if not self.hand and len(self.moskaGame.deck) == 0: # If the player doesn't have a hand and there are no cards left
                self.rank = len(self.moskaGame.get_players_condition(cond = lambda x : x.rank is not None)) + 1
        return self.rank
        
    def _fold(self):
        self.moskaGame.deck.add(self.hand.pop_cards())
        


class MoskaPlayerThreadedBase(MoskaPlayerBase):
    thread : threading.Thread = None
    moskaGame : MoskaGameThreaded = None
    
    def start(self):
        #self.thread = threading.Thread(target=self._continuous_play,name=self.name)
        self.thread = threading.Thread(target=self._continuous_play,name=self.name)
    
    def _continuous_play(self):
        print(f"{self.name} started playing...",flush=True)
        time.sleep(0.01)
        initiating_player = None
        while self.rank is None:
            with self.moskaGame.main_lock as ml:
                #current_initiating_player = self.moskaGame.get_initiating_player().pid
                #if current_initiating_player is not initiating_player:
                #    initiating_player = current_initiating_player
                #    initiated = False
                initiated = False
                if self.ready and not self.moskaGame.get_active_player() is self:
                    continue
                try:
                    self.ready = True
                    print(f"{self.name} playing...",flush=True)
                    print([pl.ready for pl in self.moskaGame.players],flush=True)
                    if len(self.moskaGame.cards_to_fall + self.moskaGame.fell_cards) > 0:
                        initiated = True
                    if len(self.moskaGame.get_players_condition(lambda x : x.rank is None)) <= 1:
                        break
                    if self.moskaGame.get_active_player() is self:
                        cards_on_table = len(self.moskaGame.cards_to_fall)
                        self._play_fall_card_from_hand()
                        now_cards_on_table = len(self.moskaGame.cards_to_fall)
                        if cards_on_table == now_cards_on_table and all([pl.ready for pl in self.moskaGame.get_players_condition(lambda x : x.rank is None)]):
                            print(f"All players have played the desired cards. Ending {self.name} turn.", flush=True)
                            self._end_turn()
                    elif self.moskaGame.get_initiating_player() is self and not initiated:
                        self._play_initial()
                        initiated = True
                    else:
                        self._play_to_target()
                except AssertionError as msg:
                    print(msg, flush=True)
                print(self.moskaGame,flush=True)
                if self.rank is not None:
                    self._end_turn()
            time.sleep(0.0000000000001)
        print(f"{self.name} finished as {self.rank}",flush=True)
        return
            
        
        