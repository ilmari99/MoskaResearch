from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Dict, List
from collections import Counter
from .Deck import Card
from ..Player.AbstractPlayer import AbstractPlayer
if TYPE_CHECKING:
    from .Game import MoskaGame
from . import utils


class _PlayToPlayer:
    """ This is the class of plays, that players can make, when they play cards to someone else or to themselves.    
    """
    player : AbstractPlayer = None
    target : AbstractPlayer = None
    cards : List[Card] = []
    moskaGame : MoskaGame = None
    def __init__(self,moskaGame : MoskaGame):
        """ Initialize
        Args:
            moskaGame (MoskaGame): MoskaGame -instance
            player (AbstractPlayer): AbstractPlayer -instance
        """
        self.moskaGame = moskaGame
        
        
    def check_cards_available(self) -> bool:
        """ Check that the cards are playable.
        Return whether the player has the play_cards in hand.
        """
        return all([card in self.player.hand.cards for card in self.cards])
    
    def check_fits(self) -> bool:
        """ Check that the cards fit in the table.
        Return whether the cards fit to the table.
        """
        return len(self.target.hand) - len(self.moskaGame.cards_to_fall) >= len(self.cards)
    
    def check_target_active(self,tg : AbstractPlayer) -> bool:
        """ Check that the target is the active player """
        return tg is self.moskaGame.get_target_player()
    
    def play(self) -> None:
        """Play the play_cards to the table;
        Modify the players hand, add cards to the table, and draw cards from the deck.
        """
        self.player.hand.pop_cards(lambda x : x in self.cards) # Remove the played cards from the players hand
        self.moskaGame.add_cards_to_fall(self.cards)           # Add the cards to the cards_to_fall -list
        self.moskaGame.glog.info(f"{self.player.name} played {self.cards} to {self.moskaGame.get_target_player().name}")
        if self.player is not self.target:
            self.player.plog.debug(f"Drew {6 - len(self.player.hand)} cards from deck")
            self.player.hand.draw(6 - len(self.player.hand))                 # Draw the to get 6 cards, if you are not playing to self
        
        

class InitialPlay(_PlayToPlayer):
    """ The play that must be done, when it is the players turn to play cards to a target (and only then)"""
    
    def __call__(self,player : AbstractPlayer, target : AbstractPlayer, cards : List[Card]):
        """This is called when the instance is called with brackets.
        Performs checks, assigns self variables and calls the play() method of super

        Args:
            target_player (AbstractPlayer): The target for who to play
            play_cards (list): The list of cards to play to the table
        """
        assert utils.check_signature([AbstractPlayer,AbstractPlayer,list],[player,target,cards]), "Incorrect input signature"
        self.player = player
        self.target = target
        self.cards = cards
        assert self.player is self.moskaGame.get_initiating_player(), f"Player is not the initiating player."
        assert len(self.moskaGame.cards_to_fall) + len(self.moskaGame.fell_cards) == 0, "The game is already initiated"
        assert self.check_single_or_multiple(), "Selected values could not be played. Only pairs or greater, cards of same values can be played."
        assert self.check_cards_available(), "Some of the played cards are not available"
        assert self.check_fits(), "Attempted to play too many cards."
        assert self.check_target_active(self.target), "Target is not active"
        self.play()
        
    def check_single_or_multiple(self):
        c = Counter([c.value for c in self.cards])
        return len(self.cards) == 1 or all((count >= 2 for count in c.values()))
        
        
class PlayToOther(_PlayToPlayer):
    """ This is the play, that players can constantly make when playing cards to an opponent after the initial play."""
    def __call__(self, player : AbstractPlayer, target : AbstractPlayer, cards : List[Card]):
        """This method is called when this instance is called with brackets.

        Args:
            target_player (AbstractPlayer): The target for who to play
            play_cards (list): The cards to play
        """
        assert utils.check_signature([AbstractPlayer,AbstractPlayer,list],[player,target,cards]), "Incorrect input signature"
        self.player = player
        self.target = target
        self.cards = cards
        assert self.check_cards_available(), "Some of the played cards are not available"
        # If player is not the target (playing to self), then the cards must fit, if playing to self, there must be deck left
        if self.player is not target:
            assert self.check_fits(), "Attempted to play too many cards."
        else:
            assert self.check_deck_left(), "There is no deck left, and playing to self is not possible."
        assert self.check_in_table(), "Some of the cards you tried to play, are not playable, because they haven't yet been played by another player."
        self.play()
        
    def check_deck_left(self):
        """ Returns True if there is still deck left. Else False """
        return len(self.player.moskaGame.deck) > 0
    
    def _playable_values(self):
        """ Return a set of values, that can be played to target"""
        return set([c.value for c in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards])
    
    def check_in_table(self):
        """ Check that the cards have already been played by either the player or an opponent"""
        playable_values = self._playable_values()
        return all((card.value in playable_values for card in self.cards))

class PlayToSelfFromDeck(_PlayToPlayer):
    def __call__(self, player : AbstractPlayer, target : AbstractPlayer, cards : List[Card]):
        assert utils.check_signature([AbstractPlayer,AbstractPlayer,list],[player,target,cards]), "Incorrect input signature"
        self.player = player
        self.target = target
        self.cards = cards
        assert self.player is self.target, "Player is not playing to self"
        assert self.check_cards_available(), "Some of the played cards are not available"
        assert self.check_target_active(self.target), "The specified target is not active"
        self.play()

class PlayToSelf(PlayToOther):
    pass


class PlayFallFromHand:
    """ A class, that is used when playing cards from the hand, to fall cards on the table"""
    moskaGame : MoskaGame = None
    player : AbstractPlayer = None
    def __init__(self,moskaGame : MoskaGame):
        """ Initialize the instance """
        self.moskaGame = moskaGame

    def __call__(self,player : AbstractPlayer, play_fall : Dict[Card,Card]):
        assert utils.check_signature([AbstractPlayer,dict],[player,play_fall]), "Incorrect input signature"
        self.player = player
        self.play_fall = play_fall
        assert self.check_cards_fall(), "Some of the played cards were not matched to a correct card to fall."
        assert self.check_player_has_turn(), "The player does not have the turn."
        assert self.check_cards_available(), "Some of the played cards are not available"
        self.play()
        
    def check_cards_fall(self):
        """Returns whether all the pairs are correctly played"""
        return all([utils.check_can_fall_card(pc,fc,self.moskaGame.triumph) for pc,fc in self.play_fall.items()])
    
    def check_cards_available(self) -> bool:
        """ Check that the cards are playable.
        Return whether the player has the play_cards in hand.
        """
        return all([card in self.player.hand.cards for card in self.play_fall.keys()])
    
    def check_player_has_turn(self):
        return self.moskaGame.get_target_player() is self.player
        
    def play(self):
        """For played_card, fall_card -pair, modify the game state;
        Remove cards to fall from table and add them to fell_cards.
        Remove the played cards from hand.
        """
        for pc,fc in self.play_fall.items():
            self.moskaGame.glog.info(f"{self.player.name} falling {pc}:{fc}")
            self.moskaGame.cards_to_fall.pop(self.moskaGame.cards_to_fall.index(fc))        # Remove from cards_to_fall
            self.moskaGame.fell_cards.append(fc)                                            # Add to fell cards
            self.player.hand.pop_cards(cond = lambda x : x == pc)                           # Remove card from hand
            self.moskaGame.fell_cards.append(pc)                                            # Add card to fell cards
        
            
class PlayFallFromDeck:
    """ Koplaus"""
    moskaGame : MoskaGame = None
    fall_method : Callable = None
    card : Card = None
    def __init__(self,moskaGame : MoskaGame):
        self.moskaGame = moskaGame
    
    def __call__(self, player : AbstractPlayer, fall_method : Callable):
        """ fall_method must accept one argument: the card that was drawn,
        and return an indexable with two values: The played card, and the card which should be fallen"""
        assert utils.check_signature([AbstractPlayer,Callable],[player,fall_method]), "Incorrect input signature"
        self.fall_method = fall_method
        self.player = player
        assert self.player is self.moskaGame.get_target_player(), "The player can't play from deck, since they are not the target."
        assert self.check_not_already_kopled(), "There is already a kopled card on the table"
        assert self.fall_method is not None, "No fall_method specified"
        assert self.check_cards_on_table(), "There are no cards on the table which should be fell"
        assert self.check_cards_in_deck(), "There are no cards from which to draw"
        self.play_card()
    
    def check_cards_on_table(self):
        """ Check if there are un-fallen cards on the table"""
        return bool(self.moskaGame.cards_to_fall)
    
    def check_cards_in_deck(self):
        """ Check that there are cards on the table, from which to draw"""
        return len(self.moskaGame.deck) > 0
    
    def check_not_already_kopled(self):
        return not any((c.kopled for c in self.moskaGame.cards_to_fall))
    
    def play_card(self):
        """ Pop a card from deck, if the card can fall a card on the table, use fall_method to select the card.
        If the card can't fall any card, add it to table.
        """
        self.card = self.moskaGame.deck.pop_cards(1)[0]
        self.card.kopled = True
        self.moskaGame.glog.info(f"{self.player.name} kopled {self.card}")
        if self.check_can_fall():
            play_fall = self.fall_method(self.card)
            # If an incorrect card is selected to fall, then a random card is picked.
            if not self.check_can_fall(in_=[play_fall[1]]):
                self.player.plog.error(f"The card {self.card} can not fall {play_fall[1]}. Falling a random card.")
                for card in self.moskaGame.cards_to_fall:
                    if utils.check_can_fall_card(self.card, card, self.moskaGame.triumph):
                        play_fall = (self.card,card)
            self.player.plog.info(f"Playing kopled card {play_fall[0]} to {play_fall[1]}")
            self.moskaGame.glog.info(f"{self.player.name} played {play_fall[0]}:{play_fall[1]}")
            self.moskaGame.cards_to_fall.pop(self.moskaGame.cards_to_fall.index(play_fall[1]))
            self.moskaGame.fell_cards.append(play_fall[1])
            self.moskaGame.fell_cards.append(play_fall[0])
        else:
            self.player.plog.debug(f"Adding {self.card} to cards_to_fall")
            self.moskaGame.glog.info(f"Adding {self.card} to cards_to_fall")
            self.moskaGame.add_cards_to_fall([self.card])
    
    def check_can_fall(self,in_ = None):
        """ Return if the card can fall a card on the table """
        in_ = self.moskaGame.cards_to_fall if not in_ else in_
        return any([utils.check_can_fall_card(self.card,fc,self.moskaGame.triumph) for fc in in_])

class EndTurn:
    """ Class representing ending a turn. """
    moskaGame : MoskaGame = None
    player : AbstractPlayer = None
    pick_cards : List[Card] = []
    def __init__(self,moskaGame : MoskaGame):
        self.moskaGame = moskaGame
    
    def __call__(self,player : AbstractPlayer, pick_cards : List[Card] = []):
        """Called at the end of a turn. Pick selected cards.

        Args:
            pick_cards (list, optional): _description_. Defaults to [].
        """
        assert utils.check_signature([AbstractPlayer,list],[player,pick_cards]), "Incorrect input signature"
        self.player = player
        self.pick_cards = pick_cards
        assert self.check_has_played_cards(), "There are no played cards, and hence the turn cannot be ended yet."
        if not pick_cards:
            assert self.check_can_pick_none() or self.check_finished(), "There are cards on the table, and they must fall or be lifted."
        else:
            assert self.check_pick_all_cards() or self.check_pick_cards_to_fall(), f"Either pick all cards that have not been fallen, or pick all cards from table"
            assert self.check_turn(), "It is not this players turn to lift the cards"
        self.pick_the_cards()
    
    def clear_table(self):
        # Only remove cards from game, if they were not picked. Then they were lifted by the player
        if self.check_pick_cards_to_fall():
            self.moskaGame.card_monitor.remove_from_game(self.moskaGame.fell_cards)
        self.moskaGame.cards_to_fall.clear()
        self.moskaGame.fell_cards.clear()
        
    def check_can_pick_none(self):
        """ Check if there are cards to fall"""
        return len(self.moskaGame.cards_to_fall) == 0
    
    def check_finished(self):
        return self.player.rank is not None
    
    def check_turn(self):
        return self.moskaGame.get_target_player() is self.player
    
    def check_pick_cards_to_fall(self):
        """ Check if every pick_card equals cards_to_fall"""
        #return all([card in self.moskaGame.cards_to_fall for card in self.pick_cards])
        #return all([picked == card for picked,card in zip(self.pick_cards,self.moskaGame.cards_to_fall)])
        return set(self.pick_cards) == set(self.moskaGame.cards_to_fall)
    
    def check_pick_all_cards(self):
        """ Check if every pick_card is in either cards_to_fall or in fell_cards and not every card is in cards_to_fall"""
        #return all([card in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards for card in self.pick_cards]) and not self.check_pick_cards_to_fall()
        #return all([picked == card for picked,card in zip(self.pick_cards,self.moskaGame.cards_to_fall + self.moskaGame.fell_cards)])
        return set(self.pick_cards) == set(self.moskaGame.cards_to_fall + self.moskaGame.fell_cards)
    
    def check_has_played_cards(self):
        """ Check if there are cards that are not fallen or that are fallen"""
        return bool(self.moskaGame.cards_to_fall + self.moskaGame.fell_cards)
    
    def pick_the_cards(self):
        """ End the turn by picking selected cards, drawing from the deck to fill hand,
        Turn the TurnCycle instance once if no cards picked, twice else
        """
        self.player.hand.cards += self.pick_cards
        self.player.hand.draw(6 - len(self.player.hand))
        for card in self.player.hand.cards:
            card.kopled = False
        self.moskaGame.turnCycle.get_next_condition(cond = lambda x : x.rank is None)
        self.moskaGame.glog.info(f"{self.player.name} ending turn.")
        self.player.plog.info(f"Lifted cards {self.pick_cards}")
        self.moskaGame.glog.info(f"{self.player.name} lifted {self.pick_cards}")
        if len(self.pick_cards) > 0 or self.player.rank is not None:
            self.moskaGame.turnCycle.get_next_condition(cond = lambda x : x.rank is None)
        self.clear_table()
        
class Skip:
    moskaGame : MoskaGame = None
    def __init__(self, moskaGame : MoskaGame):
        self.moskaGame = moskaGame
    
    def __call__(self, player : AbstractPlayer):
        assert utils.check_signature([AbstractPlayer],[player]), f"Incorrect input signature: {[player]}"
        self.player = player
        assert not self.check_is_initiating() or self.check_initiated(), "The game must be initiated, and skipping is not possible."
        assert not self.check_target_must_end_turn(), "There are no plays left, and the turn must be ended."
    
    def check_is_initiating(self):
        return self.player is self.moskaGame.get_initiating_player()
    
    def check_initiated(self):
        return len(self.moskaGame.cards_to_fall) + len(self.moskaGame.fell_cards) > 0
    
    def check_target_must_end_turn(self):
        if self.player is self.moskaGame.get_target_player():
            return self.player._must_end_turn()
        return False
    
    
    
    
    
        