from __future__ import annotations
from typing import TYPE_CHECKING, List
from .Deck import Card
if TYPE_CHECKING:   # False at runtime, since we only need MoskaGame for typechecking
    from .Game import MoskaGame, MoskaGameThreaded
from .Hand import MoskaHand
from . import utils
from .Turns import PlayFallCardFromHand, PlayFallFromDeck, PlayToOther, InitialPlay, EndTurn
import threading
import time


class MoskaPlayerBase:
    """ The base class of a moska player. This by itself is deprecated. This should not be subclassed by it self.
    To create custom play styles, one should instead subclass MoskaPlayerThreadedBase -class.
    """
    hand = None
    pid = 0
    moskaGame = None
    rank = None
    thread = None
    name = ""
    ready = False
    def __init__(self,moskaGame : MoskaGame, pid : int = 0, name : str = ""):
        """ Initialize MoskaPlayerBase -version. This by itself is a deprecated class, and the MoskaPlayerThreadedBase should be used for creating custom play styles.
        Here we initialize the distinct possible plays from Turns.py.
        
        Do not overwrite methods prefixed with "_"
        
        IMPORTANT: When subclassing, DO NOT:
        - Modify the active MoskaGame -instance in any method
        - Modify the players hand!!!! Always when seeing which cards to use from self.hand, use a COPY of the hand with eq. chand = self.hand.copy()
        - Modify the state of other players
        
        This will very likely lead to problems in the game. Looking at the hand, and getting the values is necessary to make play decisions.
        However modifying the state of the game is not necessary, since all modifications are made implicitly in the Turns -classes.
        Refer to the documentation of functions that are not prefixed with "_" for instructions on how to succesfully overwrite these methods.

        Args:
            moskaGame (MoskaGame): The MoskaGame instance in which the player is participating.
            pid (int, optional): The ID if the player. Defaults to 0. For future use.
            name (str, optional): Name of the player. Defaults to "P{pid}".
        """
        self.moskaGame = moskaGame
        self.hand = MoskaHand(moskaGame)
        self.pid = pid
        self.name = name if name else f"P{str(pid)}"
        self._playFallCardFromHand = PlayFallCardFromHand(self.moskaGame,self)
        self._playFallFromDeck = PlayFallFromDeck(self.moskaGame)
        self._playToOther = PlayToOther(self.moskaGame,self)
        self._initialPlay = InitialPlay(self.moskaGame,self)
        self._endTurn = EndTurn(self.moskaGame,self)
        
        
    def _set_pid(self,pid) -> None:
        """ Set the players pid. Currently no use."""
        self.pid = pid
    
    def _playable_values_to_table(self):
        """ Return a set of integer values that can be played to the table. This equals the set of values, that have been played to the table."""
        return set([c.value for c in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards])
    
    def _playable_values_from_hand(self):
        """ Return a set of values, that can be played to target. This is the intersection of values between the values in the table, and in the players hand."""
        return self._playable_values_to_table().intersection([c.value for c in self.hand])
    
    def _fits_to_table(self):
        """ Return the number of cards playable to the active/target player. This equals the number of cards in the targets hand, minus the number of (unfallen) cards on the table."""
        target = self.moskaGame.get_target_player()
        return len(target.hand) - len(self.moskaGame.cards_to_fall)
    
    
    def _play_to_target(self):
        """ This method is invoked to play the cards, chosen in 'play_to_target' """
        play_cards = self.play_to_target()
        target = self.moskaGame.get_target_player()
        self._playToOther(target,play_cards)
    
    
    def _play_initial(self):
        """ This function is called, when self is the initiating player, and gets to play to an empty table."""
        target = self.moskaGame.get_target_player()
        play_cards = self.play_initial()
        self._initialPlay(target,play_cards)
    
    
    
    def _can_end_turn(self):
        """ Return True if the player CAN end their turn now. Which is true, when all the other players are ready"""
        return all((pl.ready for pl in self.moskaGame.players if (pl is not self) and (pl.rank is None)))
    
    
    def _will_end_turn(self) -> bool:
        """Return True if the player must end their turn (all players ready and no playable cards in hand)
        or they want_to_end_turn  and _can_end_turn.
        
        Player must end their turn if:
        - They have no playable cards, all players are ready, and (there is no deck left or the player doesn't want to koplata)
        
        This method calls the overwritable methods 'want_to_play_from_deck' and 'want_to_end_turn'.
        This method is called all the time, when the player is the target player.
        If this method returns False, then the player plays from either deck, from hand, (or to self in the future)

        Returns:
            bool: Whether the player will end their turn
        """
        playable = 0
        for pc in self.hand:
            for fc in self.moskaGame.cards_to_fall:
                if utils.check_can_fall_card(pc,fc,self.moskaGame.triumph):
                    playable += 1
        not_ready_players = len(self.moskaGame.get_players_condition(cond = lambda x : (not x.ready) and (x.rank is None) and (x is not self)))
        #print(f"{playable} playable cards, and {not_ready_players} players who are not ready.", flush=True)
        # If there are no cards that can be played, all the players are ready, and the deck is empty
        if not self.moskaGame.cards_to_fall and not_ready_players == 0:
            return True
        # If there are no playable cards in hand, all the players are ready, and there is no deck from which to koplata
        if playable == 0 and not_ready_players == 0 and len(self.moskaGame.deck) == 0:
                return True
        # If there are no playable cards, all the players are ready and the player doesn't want to koplata
        if playable == 0 and not_ready_players == 0 and not self.want_to_play_from_deck():
            return True
        # If the player wants to end turn and the player can end their turn
        if self.want_to_end_turn() and self._can_end_turn():
            return True
        return False
    
    def _play_fall_cards(self) -> None:
        """ When the player is the target, determine which play the player wants to make next: Play from hand, play from deck (or play to self in future)"""
        ## TODO: Add playing to self as an option
        if self.want_to_play_from_deck():
            self._play_fall_from_deck()
        else:
            self._play_fall_card_from_hand()
        return
    

    def _play_fall_from_deck(self):
        """ This method is called, when the player decides to koplata. """
        self._playFallFromDeck(fall_method=self.deck_lift_fall_method)

    def _play_fall_card_from_hand(self):
        """ This method is called, when the player has decided to play cards from their hand."""
        play_cards = self.play_fall_card_from_hand()
        self._playFallCardFromHand(play_cards)
    
    
    def _end_turn(self) -> bool:
        """Called when the player must or wants to and can end their turn.

        Returns:
            bool: True if cards were picked, false otherwise
        """
        pick_cards = self.end_turn()
        self._endTurn(pick_cards)
        return bool(pick_cards)
        
    def _set_rank(self):
        """Set the players rank. Rank is None, as long as the player is still in the game.
        This is called after each turn.
        """
        if self.rank is None:   # if the player hasn't already finished
            if not self.hand and len(self.moskaGame.deck) == 0: # If the player doesn't have a hand and there are no cards left
                self.rank = len(self.moskaGame.get_players_condition(cond = lambda x : x.rank is not None)) + 1
        return self.rank
    
    @utils.check_new_card
    def _play_turn(self) -> None:
        """The basic choice selector. If the player is ready, will return instantly, except
        if the player is the current target, to accomodate playing to self and ending turn.
        Everytime after playing this, the player will be marked as ready.
        
        If the player is target:
            - Checks if the player _will_end_turn and end it if true.
            - Else attempts to fall cards.
        - If player finishes, then ends the turn.
        
        If the player is not the target:
            - If there are no cards on the table, and the player is the initiating player, then _play_initial
            - If there are cards on the table, attempts to play more cards to the target.
            
        If new values were played, sets all players ready status to False
        """
        # If the player has already played the desired cards, and he is not the target
        # If the player is the target, he might not want to play all cards at one turn, since others can then put same value cards to the table
        if self.ready and self is not self.moskaGame.get_target_player():
            return
        self.ready = True
        # If there are cards on the table; the game is already initiated
        initiated = (len(self.moskaGame.cards_to_fall) + len(self.moskaGame.fell_cards)) != 0
        # If player is the target
        if self is self.moskaGame.get_target_player():
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
        if self.rank is not None and self is self.moskaGame.get_target_player():
            self._end_turn()
        return
    
    def end_turn(self) -> List[Card]:
        """Return which cards you want to pick from the table when finishing your turn.
        Default: pick all cards that cannot be fallen.

        Returns:
            list: List of cards to pick from the table
        """
        pick_cards = self.moskaGame.cards_to_fall
        return pick_cards
    
    def play_fall_card_from_hand(self):
        """Return a dictionary of card_in_hand : card_in_table -pairs, denoting which card is used to fall which card on the table.
        This function is called when the player has decided to play from their hand.
        
        Default: Play all cards that can be played from hand, to the smallest values in the table.

        Returns:
            _type_: _description_
        """
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
    
    def deck_lift_fall_method(self, deck_card : Card):
        """A function to determine which card will fall, if a random card from the deck is lifted.
        Function should take a card -instance as argument, and return a pair (card_from_deck , card_on_table) in the same order,
        determining which card on the table to fall.
        
        This function is called, when the player decides to koplata and the koplattu card can fall a card on the table.
        If the koplattu card can't fall any card on the table, then the card is just placed on the table.

        Args:
            deck_card (Card): The lifted card from the deck

        Returns:
            tuple(Card,Card): The input card from deck, the card on the table.
        """
        for card in self.moskaGame.cards_to_fall:
            if utils.check_can_fall_card(deck_card,card,self.moskaGame.triumph):
                return (deck_card,card)
            
    def want_to_end_turn(self):
        """ Return True if the player (as target) wants to premeturely end the turn by calling the _end_turn() method 
        which lifts the cards specified in end_turn()
        
        Default: Want to end turn, when there are no cards to fall left"""
        if not self.moskaGame.cards_to_fall:
            return True
        else:
            False 
            
    def play_initial(self):
        """ Return a list of cards that will be played to target on an initiating turn. AKA playing to an empty table.
        Default: Play all the smallest cards in hand, that fit to table."""
        min_card = min([c.value for c in self.hand])
        hand = self.hand.copy()
        play_cards = hand.pop_cards(cond=lambda x : x.value == min_card,max_cards = self._fits_to_table())
        return play_cards
    
    def want_to_play_from_deck(self):
        """ When you are the target, return True if you want to play from deck (koplata) or else False.
        Default: Play from deck if there is only one card left (which is always triumph). This however is not always the best choice."""
        if len(self.moskaGame.deck) == 1:
            return True
        return False
    
    def play_to_target(self) -> List[Card]:
        """ Return a list of cards, that will be played to target.
        This function is called, when there are cards on the table, and you can play cards to a target
        
        This method is meant to be overwriteable.
        Default: Play all playable values that fit.
        """
        playable_values = self._playable_values_from_hand()
        play_cards = []
        if playable_values:
            hand = self.hand.copy()     # Create a copy of the game.
            play_cards = hand.pop_cards(cond=lambda x : x.value in playable_values,max_cards = self._fits_to_table())
        return play_cards
            

class MoskaPlayerThreadedBase(MoskaPlayerBase):
    """Subclassing this class, and writing new methods for:
    play_to_target() -> List of cards to play to a target
    play_initial() -> List of cards to play to a target, when you are initiating the turn
    deck_lift_fall_method(card_from_deck) -> (card_from_deck, to_which_card_on_table_you_want_to_play)
    play_fall_card_from_hand() -> Return a dictionary of values {card_from_hand : card_on_table}
    end_turn() -> List of cards that you want to pick from the table, when ending your turn.
    """
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