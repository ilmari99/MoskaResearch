from __future__ import annotations
import itertools
import random
from typing import TYPE_CHECKING, List, Tuple

from Moska.Player.AbstractPlayer import AbstractPlayer
from .Deck import Card, StandardDeck
from .utils import check_can_fall_card
if TYPE_CHECKING:
    from .Game import MoskaGame
    


class CardMonitor:
    """
    CardMonitor is a class that keeps track of the cards that are known to each player, and which cards have fallen.
    """
    
    
    player_cards : dict[str,List[Card]]= {}
    cards_fall_dict : dict[Card,List[Card]] = {}
    game : MoskaGame = None
    started : bool = False
        
    def __init__(self,moskaGame : MoskaGame):
        self.game = moskaGame
        self.player_cards = {}
        self.cards_fall_dict = {}
        self.started = False
        
    def start(self) -> None:
        """ Start tracking cards.
        This can only be done right when the game starts, as for ex the triumph card is not known until then.
        """
        if self.started:
            return
        # 
        for pl in self.game.players:
            self.player_cards[pl.name] = []
            self.update_known(pl.name,cards=[Card(-1,"X") for _ in range(6)],add=True)
            if len(self.player_cards[pl.name]) != 6:
                self.player_cards.pop(pl.name)
        self.game.glog.info(f"Created card monitor. Player Cards:")
        for pl, cards in self.player_cards.items():
            self.game.glog.info(f"{pl} : {cards}")
        self.make_cards_fall_dict()
        assert len(self.cards_fall_dict) == 52, f"Invalid make cards fall dict"
        self.started = True
        self.game.glog.info(f"Card monitor started")
        return
    
    def get_cards_possibly_in_deck(self,player : AbstractPlayer = None) -> set[Card]:
        """Get a list of cards that are possibly in the deck. This is done by checking which cards are not known to the player.
        """
        # If there is only one card left in the deck, it is the triumph card
        if len(self.game.deck) == 1:
            return [self.game.triumph_card]
        if len(self.game.deck) == 0:
            return []
        cards_not_in_deck = self.game.fell_cards + self.game.cards_to_fall
        if player is not None:
            cards_not_in_deck += player.hand.copy().cards
        for pl, cards in self.player_cards.items():
            if player is not None and pl == player.name:
                continue
            for card in cards:
                if card != Card(-1,"X"):
                    cards_not_in_deck.append(card)
        #self.player.plog.debug(f"Cards NOT in deck: {len(cards_not_in_deck)}")
        cards_possibly_in_deck = set(self.game.card_monitor.cards_fall_dict.keys()).difference(cards_not_in_deck)
        #self.player.plog.debug(f"Cards possibly in deck: {len(cards_possibly_in_deck)}")
        return cards_possibly_in_deck
    
    def get_sample_cards_from_deck(self,player : AbstractPlayer,ncards : int, max_samples : int = 100) -> List[Card]:
        """Get a list of tuples of ncards, where each tuple is a unique sample of cards from the deck.
        """
        assert ncards <= 6, f"Cannot sample more than 6 cards"
        assert ncards > 0, f"Cannot sample less than 1 card"
        cards_possibly_in_deck = self.get_cards_possibly_in_deck(player)
        # If there are less cards in the deck than ncards, return all cards
        if len(cards_possibly_in_deck) < ncards:
            return tuple(cards_possibly_in_deck)
        samples = set()
        # Get different card combinations
        combs = itertools.combinations(cards_possibly_in_deck,ncards)
        for comb in combs:
            samples.add(comb)
            if len(samples) >= max_samples:
                break
        return list(samples)
        
    def make_cards_fall_dict(self):
        """Create the cards_fall_dict by going through each card and checking if each card can be fell with the card
        """
        deck = StandardDeck().pop_cards(52)
        for i,card in enumerate(deck):
            self.cards_fall_dict[card] = []
            for i2,card2 in enumerate(deck):
                if i == i2:
                    continue
                # Requires the game to be started, otherwise we have no information on the triumph suit
                if check_can_fall_card(card,card2,self.game.triumph):
                    self.cards_fall_dict[card].append(card2)
        return
    
    def update_from_move(self, moveid : str, args : Tuple) -> None:
        """Update the CardMonitor instance, given moveid and the arguments passed to MoskaGame

        Args:
            moveid (str): A string identifying the move
            args (Tuple): Arguments for the move

        Returns:
            None
        """
        player = args[0]
        if moveid == "EndTurn":
            picked = args[1]
            self.update_known(player.name,picked,add=True)
        elif moveid in ["InitialPlay", "PlayToOther", "PlayToSelf"]:
            played = args[-1]
            self.update_known(player.name,played,add=False)
        elif moveid == "PlayFallFromHand":
            played = list(args[-1].keys())
            #fell = list(args[-1].values())
            self.update_known(player.name,played,add=False)
            # No need to remove from game yet, since the cards might be lifted
        # After updating known cards, check if the player lifted unknown cards from deck
        self.update_unknown(player.name)
        # No updates to hand when playing from deck or skipping
        return
    
    def remove_from_game(self,cards : List[Card]) -> None:
        """ Remove cards from the card monitor. Both as keys and values in the cards_fall_dict.
        This is called at the end of a turn, and when cards are fallen from hand.
        
        Called from Turns.EndTurn.clear_table with moskaGame.fell_cards IF all cards were not lifted
        Args:
            cards (List[Card]): _description_
        """
        # Remove the removed cards from the cards_fall_dict
        for card in cards:
            # Remove the fallen card as a key
            if card in self.cards_fall_dict:
                self.cards_fall_dict.pop(card)
                #self.game.glog.debug(f"Removed {card} from cards_fall_dict keys")
        # Remove the card as value from the list
        for card_d, falls in self.cards_fall_dict.copy().items():
            for card in cards:
                if card in falls:
                    self.cards_fall_dict[card_d].remove(card)
                    #self.game.glog.debug(f"Removed {card} from {card_d} list of cards")
        for card,falls in self.cards_fall_dict.items():
            pass#self.game.glog.info(f"{card} : {len(falls)}")
        return
        
    
    def update_unknown(self, player_name : str) -> None:
        """Update missing cards from the players hand. Either add or remove unknown cards (Card(-1,"X"))

        Args:
            player_name (str): Players name
        """
        # The cards we know the player has
        known_cards = self.player_cards[player_name]
        # Get the cards the player actually has
        actual_cards = self.game.get_players_condition(cond = lambda x : x.name == player_name)[0].hand.cards
        missing = len(actual_cards) - len(known_cards)
        add = True
        if missing < 0:
            add = False
            #raise ValueError(f"Knowing more cards in a players hand than there are cards is impossible!!")
        if missing == 0:
            return
        # Either add unknown cards, or remove unknown cards from the players hand
        self.update_known(player_name,[Card(-1,"X") for _ in range(abs(missing))],add=add)
        return
    
    def update_known(self, player_name : str, cards : List[Card], add : bool = False) -> None:
        """ Update KNOWN cards from the player by either adding or removing cards from their hand. This is also used to
        add unknown cards, since we still know they forexample lifted a certain number of cards.

        Args:
            player_name (_type_): The players name
            cards (_type_): List of cards to add or remove from the player shand
            add (bool, optional): Whether to add (True) or remove (False) cards from the players hand. Defaults to False.
        """
        # If we are removing cards from the players hand
        if not add:
            for card in cards:
                # If the card we want to remove from the players hand is not known, then mark it as an unknown card
                if card not in self.player_cards[player_name]:
                    card = Card(-1,"X")
                #self.game.glog.info(f"CardMonitor: Removed {card} from {player_name}")
                try:
                    self.player_cards[player_name].remove(card)
                except:
                    print(f"CardMonitor: Tried to remove {card} from {player_name}, but it was not in the players hand")
                    raise ValueError(f"CardMonitor: Tried to remove {card} from {player_name}, but it was not in the players hand")
        # If we want to add cards to the players hand
        else:
            self.player_cards[player_name] = cards + self.player_cards[player_name]
            #self.game.glog.info(f"CardMonitor: Added {cards} to {player_name}")
        # Print the players cards
        #for pl, cards in self.player_cards.items():
        #    self.game.glog.debug(f"{pl} : {cards}")
        #self.game.glog.debug("\n")
        return