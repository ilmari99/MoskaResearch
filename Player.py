from __future__ import annotations
import itertools
import multiprocessing
from typing import TYPE_CHECKING
from copy import copy, deepcopy
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame fro typechekcing
    from Game import MoskaGame, MoskaGameThreaded, MoskaGameMultiprocess
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
    
    
    def _play_to_target(self):
        """ This method is invoked to play the cards, chosen in 'play_to_target'"""
        play_cards = self.play_to_target()
        target = self.moskaGame.get_active_player()
        self._playToOther(target,play_cards)
    
    def play_to_target(self):
        """ Return a list of cards, that will be played to target. 
        Default: Play all playable values that fit."""
        playable_values = self._playable_values_from_hand()
        play_cards = []
        if playable_values:
            hand = self.hand.copy()
            play_cards = hand.pop_cards(cond=lambda x : x.value in playable_values,max_cards = self._fits_to_table())
        return play_cards
    
    def _play_initial(self):
        target = self.moskaGame.get_active_player()
        play_cards = self.play_initial()
        self._initialPlay(target,play_cards)
    
    
    def play_initial(self):
        """ Return a list of cards that will be played to target on an initiating turn.
        Default: Play all the smallest cards in hand, that fit to table."""
        min_card = min([c.value for c in self.hand])
        hand = self.hand.copy()
        play_cards = hand.pop_cards(cond=lambda x : x.value == min_card,max_cards = self._fits_to_table())
        return play_cards
    
    def want_to_play_from_deck(self):
        """ return True if you want to play from deck (koplata) or False.
        Default: Play from deck if there is only one card left (which is always triumph). This however is not always the best choice."""
        if len(self.moskaGame.deck) == 1:
            return True
        return False
    
    def _can_end_turn(self):
        """ Return True if the player CAN end their turn now. Which is true, when all the other players are ready"""
        return all((pl.ready for pl in self.moskaGame.players if (pl is not self) and (pl.rank is None)))
    
    def want_to_end_turn(self):
        """ Return True if the player (as target) wants to end the turn.
        Default: Want to end turn, when there are no cards to fall left"""
        if not self.moskaGame.cards_to_fall:
            return True
        else:
            False 
    
    def _will_end_turn(self):
        """ Return True if the player must end their turn, they want to end their turn (and can)"""
        playable = 0
        for pc in self.hand:
            for fc in self.moskaGame.cards_to_fall:
                if utils.check_can_fall_card(pc,fc,self.moskaGame.triumph):
                    playable += 1
        not_ready_players = len(self.moskaGame.get_players_condition(cond = lambda x : (not x.ready) and (x.rank is None) and (x is not self)))
        print(f"{playable} playable cards, and {not_ready_players} players who are not ready.", flush=True)
        #if self.rank is not None:
        #    return True
        # If there are no cards that can be played, all the players are ready, and the deck is empty
        if not self.moskaGame.cards_to_fall and not_ready_players == 0:
            return True
        # If there are no playable cards in hand, all the players are ready, and there is no deck from which to koplata
        if playable == 0 and not_ready_players == 0 and len(self.moskaGame.deck) == 0:
                return True
        # If there are no playable cards, all the players are ready and the player doesn't want to koplata
        if playable == 0 and not_ready_players == 0 and not self.want_to_play_from_deck():
            return True
        if self.want_to_end_turn() and self._can_end_turn():
            return True
        return False
    
    def _play_fall_cards(self):
        """ When the player is the target, determine which play the player wants to make next, and play it"""
        ## TODO: Add playing to self as an option
        if self.want_to_play_from_deck():
            self._play_fall_from_deck()
        else:
            self._play_fall_card_from_hand()
        return
    
    def deck_lift_fall_method(self, deck_card):
        """ A function to determine which card will fall, if a random card from the deck is lifted.
        Function should take a card -instance as argument, and return a pair (card_from_deck , card_on_table) in the same order,
        determining which card on the table to fall."""
        for card in self.moskaGame.cards_to_fall:
            if utils.check_can_fall_card(deck_card,card,self.moskaGame.triumph):
                return (deck_card,card)

    def _play_fall_from_deck(self):
        self._playFallFromDeck(fall_method=self.deck_lift_fall_method)

    def _play_fall_card_from_hand(self):
        play_cards = self.play_fall_card_from_hand()
        self._playFallCardFromHand(play_cards)
    
    def play_fall_card_from_hand(self):
        """ Return a dictionary of card_in_hand : card_in_table -pairs, denoting which card is used to fall which card on the table."""
        play_cards = {}
        for fall_card in self.moskaGame.cards_to_fall:
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
    
    @utils.check_new_card
    def _play_turn(self):
        # If the player has already played the desired cards, and he is not the target
        # If the player is the target, he might not want to play all cards at one turn, since others can then put same value cards to the table
        if self.ready and self is not self.moskaGame.get_active_player():
            return
        self.ready = True
        # If there are cards on the table; the game is already initiated
        initiated = (len(self.moskaGame.cards_to_fall) + len(self.moskaGame.fell_cards)) != 0
        # If player is the target
        if self is self.moskaGame.get_active_player():
            # If the player will end their turn; They must by the rules, cant/wont koplata, or they want to and can end the rule
            if self._will_end_turn():
                self._end_turn()
            # If the player doesn't want to, and doesn't have to end their turn, then play cards from either hand, or deck.
            else:
                self._play_fall_cards()
        # If the player is the initiating player, and the game has not been initiated
        elif not initiated and self is self.moskaGame.get_initiating_player():
            self._play_initial()
        # Else, if cards fit to table
        elif self._fits_to_table() > 0:
            self._play_to_target()
        self._set_rank()
        if self.rank is not None and self is self.moskaGame.get_active_player():
            self._end_turn()
        return
            
        
            
            
            
                
            
            
        
        


class MoskaPlayerThreadedBase(MoskaPlayerBase):
    thread : threading.Thread = None
    moskaGame : MoskaGameThreaded = None
    
    def start(self):
        self.thread = threading.Thread(target=self._continuous_play,name=self.name)
    
    
    def _continuous_play(self):
        print(f"{self.name} started playing...",flush=True)
        while self.rank is None:
            # Acquire the lock for moskaGame
            with self.moskaGame.main_lock as ml:
                try:
                    print(f"{self.name} playing...",flush=True)
                    print([pl.ready for pl in self.moskaGame.players],flush=True)
                    if len(self.moskaGame.get_players_condition(lambda x : x.rank is None)) <= 1:
                        break
                    self._play_turn()
                except AssertionError as msg:
                    print(msg, flush=True)
                print(self.moskaGame,flush=True)
            time.sleep(0.00001)
        print(f"{self.name} finished as {self.rank}",flush=True)
        return
    
    """
    def _continuous_play(self):
        print(f"{self.name} started playing...",flush=True)
        while self.rank is None:
            # Acquire the lock for moskaGame
            with self.moskaGame.main_lock as ml:
                initiated = False
                # If we have already played and no one has played any new cards to the table, or ended their turn since
                if self.ready and not self.moskaGame.get_active_player() is self:
                    continue
                try:
                    print(f"{self.name} playing...",flush=True)
                    print([pl.ready for pl in self.moskaGame.players],flush=True)
                    self.ready = True
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
                time.sleep(0.00001)
        print(f"{self.name} finished as {self.rank}",flush=True)
        return
        """