from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
from .Deck import Card, StandardDeck
from .utils import check_can_fall_card
if TYPE_CHECKING:
    from .Game import MoskaGame
    


class CardMonitor:
    player_cards : dict[str,List[Card]]= {}
    cards_fall_dict : dict[Card,List[Card]] = {}
    game : MoskaGame = None
    started : bool = False
        
    def __init__(self,moskaGame : MoskaGame):
        self.game = moskaGame
        self.player_cards = {}
        self.cards_fall_dict = {}
        self.started = False
        
    def start(self):
        if self.started:
            return
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
        return
        
    def make_cards_fall_dict(self):
        """Create the crds_fall_dict by going through each card and checking if each card can be fell with the card
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
    
    def update_from_move(self, moveid : str, args : Tuple) -> Tuple:
        """Update the CardMonitor instance, given moveid and the arguments passed to MoskaGame

        Args:
            moveid (str): _description_
            args (Tuple): _description_

        Returns:
            Tuple: _description_
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
            fell = list(args[-1].values())
            self.update_known(player.name,played,add=False)
            self.remove_from_game(played + fell)
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
                self.game.glog.debug(f"Removed {card} from cards_fall_dict keys")
        # Remove the card as value from the list
        for card_d, falls in self.cards_fall_dict.copy().items():
            for card in cards:
                if card in falls:
                    self.cards_fall_dict[card_d].remove(card)
                    self.game.glog.debug(f"Removed {card} from {card_d} list of cards")
        for card,falls in self.cards_fall_dict.items():
            self.game.glog.info(f"{card} : {len(falls)}")
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
        self.update_known(player_name,[Card(-1,"X") for _ in range(missing)],add=add)
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
                self.game.glog.info(f"CardMonitor: Removed {card} from {player_name}")
                self.player_cards[player_name].remove(card)
        # If we want to add cards to the players hand
        else:
            self.player_cards[player_name] += cards
            self.game.glog.info(f"CardMonitor: Added {cards} to {player_name}")
        # Print the players cards
        for pl, cards in self.player_cards.items():
            self.game.glog.debug(f"{pl} : {cards}")
        self.game.glog.debug("\n")
        return