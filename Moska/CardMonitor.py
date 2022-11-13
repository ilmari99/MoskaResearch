from __future__ import annotations
from typing import TYPE_CHECKING, List
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
        self.started = True
        return
        
    def make_cards_fall_dict(self):
        deck = StandardDeck()
        for i in range(len(deck.cards)-1):
            card = deck.cards.pop()
            for i in range(len(deck.cards)):
                card2 = deck.cards.pop()
                # Requires the game to be started, otherwise we have no information on the triumph suit
                if check_can_fall_card(card,card2,self.game.triumph):
                    if card not in self.cards_fall_dict:
                        self.cards_fall_dict[card] = []
                    self.cards_fall_dict[card].append(card2)
                deck.cards.appendleft(card2)
            deck.cards.appendleft(card)
        return
    
    def update_from_move(self, moveid, args):
        player = args[0]
        if moveid == "EndTurn":
            picked = args[1]
            self.update_known(player.name,picked,add=True)
        elif moveid in ["InitialPlay", "PlayToOther", "PlayToSelf"]:
            played = args[-1]
            self.update_known(player.name,played,add=False)
        elif moveid == "PlayFallFromHand":
            played = list(args[-1].keys())
            self.update_known(player.name,played,add=False)
        # After updating known cards, check if the player lifted unknown cards from deck
        self.update_unknown(player.name)
        # No updates to hand when playing from deck or skipping
        return
    
    def remove_from_game(self,cards : List[Card]):
        # Called from Turns.EndTurn.clear_table
        for card in cards:
            if card in self.cards_fall_dict:
                self.cards_fall_dict.pop(card)
        for card_d, falls in self.cards_fall_dict.copy().items():
            for card in cards:
                if card in falls:
                    self.cards_fall_dict[card_d].remove(card)
        self.game.glog.info(f"Cards can fall: \n")
        for card,falls in self.cards_fall_dict.items():
            self.game.glog.info(f"{card} : {len(falls)}")
        return
        
    
    def update_unknown(self, player_name):
        known_cards = self.player_cards[player_name]
        actual_cards = self.game.get_players_condition(cond = lambda x : x.name == player_name)[0].hand.cards
        missing = len(actual_cards) - len(known_cards)
        add = True
        if missing < 0:
            add = False
            #raise ValueError(f"Knowing more cards in a players hand than there are cards is impossible!!")
        if missing == 0:
            return
        self.update_known(player_name,[Card(-1,"X") for _ in range(missing)],add=add)
    
    def update_known(self, player_name, cards, add = False):
        if not add:
            for card in cards:
                if card not in self.player_cards[player_name]:
                    card = Card(-1,"X")
                self.game.glog.info(f"Removed {card} from {player_name}")
                self.player_cards[player_name].remove(card)
        else:
            self.game.glog.info(f"added {cards} to {player_name}")
            self.player_cards[player_name] += cards
            
        for pl, cards in self.player_cards.items():
            self.game.glog.info(f"{pl} : {cards}")
        self.game.glog.info("\n")