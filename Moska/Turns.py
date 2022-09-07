from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from Moska.Deck import Card
if TYPE_CHECKING:
    from .Game import MoskaGame
    from .Player import MoskaPlayer
from . import utils

class _PlayToPlayer:
    """ This is the class of plays, that players can make, when they play cards to someone else or to themselves.    
    """
    def __init__(self,moskaGame : MoskaGame, player : MoskaPlayer):
        """ Initialize
        Args:
            moskaGame (MoskaGame): MoskaGame -instance
            player (MoskaPlayer): MoskaPlayer -instance
        """
        self.moskaGame = moskaGame
        self.player = player
        
        
    def check_cards_available(self) -> bool:
        """ Check that the cards are playable.
        Return whether the player has the play_cards in hand.
        """
        return all([card in self.player.hand.cards for card in self.play_cards])
    
    def check_fits(self) -> bool:
        """ Check that the cards fit in the table.
        Return whether the cards fit to the table.
        """
        return len(self.target_player.hand) - len(self.moskaGame.cards_to_fall) >= len(self.play_cards)
    
    def check_target_active(self,tg : MoskaPlayer) -> bool:
        """ Check that the target is the active player """
        return tg is self.moskaGame.get_target_player()
    
    def play(self) -> None:
        """Play the play_cards to the table;
        Modify the players hand, add cards to the table, and draw cards from the deck.
        """
        self.player.hand.pop_cards(lambda x : x in self.play_cards) # Remove the played cards from the players hand
        self.moskaGame.add_cards_to_fall(self.play_cards)           # Add the cards to the cards_to_fall -list
        self.player.hand.draw(6 - len(self.player.hand))                 # Draw the to get 6 cards
        #self.player._set_rank()
        

class InitialPlay(_PlayToPlayer):
    """ The play that must be done, when it is the players turn to play cards to a target (and only then)"""
    def __call__(self,target_player : MoskaPlayer, play_cards : list):
        """This is called when the instance is called with brackets.
        Performs checks, assigns self variables and calls the play() method of super

        Args:
            target_player (MoskaPlayer): The target for who to play
            play_cards (list): The list of cards to play to the table
        """
        self.target_player = target_player
        self.play_cards = play_cards
        assert self.check_cards_available(), "Some of the played cards are not available"
        assert self.check_fits(), "Attempted to play too many cards."
        assert self.check_target_active(self.target_player), "Target is not active"
        self.play()
        
class PlayToOther(_PlayToPlayer):
    """ This is the play, that players can constantly make when playing cards to an opponent after the initial play."""
    def __call__(self, target_player : MoskaPlayer, play_cards : list):
        """This method is called when this instance is called with brackets.

        Args:
            target_player (MoskaPlayer): The target for who to play
            play_cards (list): The cards to play
        """
        self.target_player = target_player
        self.play_cards = play_cards
        assert self.check_cards_available(), "Some of the played cards are not available"
        assert self.check_fits(), "Attempted to play too many cards."
        assert self.check_in_table(), "Some of the cards you tried to play, are not playable, because they haven't yet been played by another player."
        self.play()
    
    def _playable_values(self):
        """ Return a set of values, that can be played to target"""
        return set([c.value for c in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards])
    
    def check_in_table(self):
        """ Check that the cards have already been played by either the player or an opponent"""
        playable_values = self._playable_values()
        return all((card.value in playable_values for card in self.play_cards))


class PlayFallCardFromHand:
    """ A class, that is used when playing cards from the hand, to fall cards on the table"""
    def __init__(self,moskaGame : MoskaGame, player : MoskaPlayer):
        """ Initialize the instance """
        self.moskaGame = moskaGame
        self.player = player

    def __call__(self,play_fall : dict):
        
        self.play_fall = play_fall
        assert self.check_cards_fall(), "Some of the played cards were not matched to a correct card to fall."
        assert self.check_player_has_turn(), "The player does not have the turn."
        self.play()
        
    def check_cards_fall(self):
        """Returns whether all the pairs are correctly played"""
        return all([utils.check_can_fall_card(pc,fc,self.moskaGame.triumph) for pc,fc in self.play_fall.items()])
    
    def check_player_has_turn(self):
        return self.moskaGame.get_target_player() is self.player
        
    def play(self):
        """For played_card, fall_card -pair, modify the game state;
        Remove cards to fall from table and add them to fell_cards.
        Remove the played cards from hand.
        """
        for pc,fc in self.play_fall.items():
            self.moskaGame.cards_to_fall.pop(self.moskaGame.cards_to_fall.index(fc))        # Remove from cards_to_fall
            self.moskaGame.fell_cards.append(fc)                                            # Add to fell cards
            self.player.hand.pop_cards(cond = lambda x : x == pc)                           # Remove card from hand
            self.moskaGame.fell_cards.append(pc)                                            # Add card to fell cards
        
            
class PlayFallFromDeck:
    """ Koplaus"""
    def __init__(self,moskaGame : MoskaGame,fall_method : Callable = None):
        self.moskaGame = moskaGame
        self.fall_method = fall_method
        
    def __call__(self, fall_method : Callable = None):
        """ fall_method must accept one argument: the card that was drawn,
        and return an indexable with two values: The played card, and the card which should be fallen"""
        if fall_method is not None:
            self.fall_method = fall_method
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
        self.card = Card(self.card.value,self.card.suit,True)
        if self.check_can_fall():
            play_fall = self.fall_method(self.card)
            self.moskaGame.cards_to_fall.pop(self.moskaGame.cards_to_fall.index(play_fall[1]))
            self.moskaGame.fell_cards.append(play_fall[1])
            self.moskaGame.fell_cards.append(play_fall[0])
        else:
            self.moskaGame.add_cards_to_fall([self.card])
            
    def check_can_fall(self):
        """ Return if the card can fall a card on the table """
        return any([utils.check_can_fall_card(self.card,fc,self.moskaGame.triumph) for fc in self.moskaGame.cards_to_fall])

class EndTurn:
    """ Class representing ending a turn. """
    def __init__(self,moskaGame : MoskaGame, player : MoskaPlayer):
        self.moskaGame = moskaGame
        self.player = player
    
    
    def __call__(self,pick_cards : list = []):
        """Called at the end of a turn. Pick selected cards.

        Args:
            pick_cards (list, optional): _description_. Defaults to [].
        """
        self.pick_cards = pick_cards
        assert self.check_has_played_cards(), "There are no played cards, and hence the turn cannot be ended yet."
        if not pick_cards:
            assert self.check_can_pick_none() or self.check_finished(), "There are cards on the table, and they must fall or be lifted."
        else:
            assert self.check_pick_all_cards() or self.check_pick_cards_to_fall()
            assert self.check_turn(), "It is not this players turn to lift the cards"
        self.pick_the_cards()
    
    def clear_table(self):
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
        return all([picked == card for picked,card in zip(self.pick_cards,self.moskaGame.cards_to_fall)])
    
    def check_pick_all_cards(self):
        """ Check if every pick_card is in either cards_to_fall or in fell_cards and not every card is in cards_to_fall"""
        #return all([card in self.moskaGame.cards_to_fall + self.moskaGame.fell_cards for card in self.pick_cards]) and not self.check_pick_cards_to_fall()
        return all([picked == card for picked,card in zip(self.pick_cards,self.moskaGame.cards_to_fall + self.moskaGame.fell_cards)])
    
    def check_has_played_cards(self):
        """ Check if there are cards that are not fallen or that are fallen"""
        return bool(self.moskaGame.cards_to_fall + self.moskaGame.fell_cards)
    
    def pick_the_cards(self):
        """ End the turn by picking selected cards, drawing from the deck to fill hand,
        Turn the TurnCycle instance once if no cards picked, twice else
        """
        self.player.hand.cards += self.pick_cards
        self.player.hand.draw(6 - len(self.player.hand))
        self.moskaGame.turnCycle.get_next_condition(cond = lambda x : x.rank is None)
        print(f"All players have played the desired cards. Ending {self.player.name} turn.", flush=True)
        print(f"Lifted cards {self.pick_cards}", flush=True)
        if len(self.pick_cards) > 0 or self.player.rank is not None:
            self.moskaGame.turnCycle.get_next_condition(cond = lambda x : x.rank is None)
        self.clear_table()
    
    
    
    
    
        