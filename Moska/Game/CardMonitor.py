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
    Each time a move is played, the CardMonitor is updated with the new information, such as:
    - If a player picked cards from the table, each player now knows them.
    - If cards are discarded from the table, the cards are removed from the cards_fall_dict as keys and values

    The attributes:
    - player_cards: A dictionary that maps player names to a list of cards. Some cards in this list are not known, and are marked as Card(-1,"X").
    If the CardMonitor has observed, that a player has publically lifted some cards, then the unknown cards are updated.
    - cards_fall_dict: A dictionary that maps cards to a list of cards. The cards in the list are the cards, that the key card can fall.
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
        This can only be done right when the game starts, as for ex the trump card is not known until then.
        """
        if self.started:
            return
        # The trump card has been changed at this point.
        # Search where the card is, and if a player has it everyone knows it, so update the card monitor
        for pl in self.game.players:
            self.player_cards[pl.name] = []

            if self.game._orig_trump_card in pl.hand.cards:
                cards = [Card(-1,"X") for _ in range(5)] + [self.game._orig_trump_card]
                self.game.glog.info(f"Trump card on player: {pl.name}")
            else:
                cards = [Card(-1,"X") for _ in range(6)]
            self.update_known(pl.name,cards=cards,add=True)
            # Old backup, if there are incorrect players
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
    
    def get_hidden_cards(self,player : AbstractPlayer) -> List[Card]:
        """Get a list of cards whose location is not known to the player.
        This has all cards in a standard deck, EXCEPT:
        - Cards that have not been publically lifted from the table
        - Cards that are in the table (fell, and not-fell cards)
        - The cards the player calling this function has in their hand.

        All other cards locations are not known
        """
        known_cards = player.hand.copy().cards
        known_cards += self.game.cards_to_fall.copy()
        known_cards += self.game.fell_cards.copy()
        for pl, cards in self.player_cards.items():
            if pl == player.name:
                continue
            known_cards += [card for card in cards if card != Card(-1,"X")]
        # Each player knows which cards are still in the game, and which cards are known to them
        # The hidden cards are the intersection of the cards that are still in the game and the cards that are known to the player
        player.plog.info(f"Known cards: {known_cards}")
        hidden_cards = [card for card in self.cards_fall_dict.keys() if card not in known_cards]
        return hidden_cards
    
    
    def get_cards_possibly_in_deck(self,player : AbstractPlayer) -> list[Card]:
        """Get a list of cards that are possibly in the deck. This is done by checking which cards are not known to the player.
        This is the same as get_hidden_cards, but with checks, for if the deck is empty, or there is only one card.
        Otherwise, all the hidden cards are returned
        """
        # If there is only one card left in the deck, it has to be the trump card
        if len(self.game.deck) == 1:
            return [self.game.trump_card]
        if len(self.game.deck) == 0:
                return []
        hidden_cards = self.get_hidden_cards(player)
        #random.shuffle(hidden_cards)
        # Just casually check that there are no duplicates (hash)
        assert len(hidden_cards) == len(set(hidden_cards)), f"Duplicates in hidden cards"
        return hidden_cards
    
    def get_sample_cards_from_deck(self,player : AbstractPlayer,ncards : int, max_samples : int = 10) -> List[Tuple[Card]]:
        """Get a list of tuples of ncards, where each tuple is a unique sample of cards from the deck.
        For example, the cards possibly in deck are [0,2,5,1,9,11], and we want 10 (max_samples) of 2 (ncards) cards.
        This function will return 10 2 card combinations from all combinations of 2 cards (randomly). since we do combinations,
        there won't be symmetrical pairs, such as (1,2) and (2,1)
        """
        assert ncards > 0, f"Cannot sample less than 1 card"
        cards_possibly_in_deck = self.get_cards_possibly_in_deck(player)
        # If there are less cards in the deck than ncards, return all cards
        if len(cards_possibly_in_deck) < ncards:
            player.plog.debug(f"Less cards possibly in deck than are lifted, returning all cards")
            return tuple(cards_possibly_in_deck)
        # if the player will not lift the trump card, we can remove it.
        if ncards < len(self.game.deck):
            cards_possibly_in_deck.remove(self.game.trump_card)
        samples = []
        # Shuffle the cards for fun
        random.shuffle(cards_possibly_in_deck)
        def random_combination(iterable, r):
            # Read all combination tuples to a list
            # Return r random combinations from all the combinations itertools.combinations(cards, ncards)
            pool = tuple(iterable)
            n = len(pool)
            # No sort needed
            indices = random.sample(range(n), min(r, n))
            return tuple(pool[i] for i in indices)
        combs = itertools.combinations(cards_possibly_in_deck,ncards)
        # If the player will lift the remaining deck, there will be the trump card in the lifted cards
        if ncards >= len(self.game.deck):
            combs = filter(lambda x: self.game.trump_card in x,combs)
        combs = random_combination(combs,max_samples)
        for comb in combs:
            samples.append(comb)
            if len(samples) >= max_samples:
                break
        player.plog.info(f"Sampled {len(samples)} cards from deck")
        player.plog.debug(f"Sampled cards: {samples}")
        return samples

    def make_cards_fall_dict(self):
        """Create the cards_fall_dict by going through each card
        and checking if each card can be fell with the card
        """
        deck = StandardDeck().pop_cards(52)
        for i,card in enumerate(deck):
            self.cards_fall_dict[card] = []
            for i2,card2 in enumerate(deck):
                if i == i2:
                    continue
                # Requires the game to be started, otherwise we have no information on the trump suit
                if check_can_fall_card(card,card2,self.game.trump):
                    self.cards_fall_dict[card].append(card2)
        return
    
    def update_from_move(self, moveid : str, args : Tuple) -> None:
        """Update the CardMonitor instance, given moveid (str) and the arguments passed to MoskaGame.
        """
        player = args[0]
        # If the move is "EndTurn", update the players public cards, based on what the player lifted
        if moveid == "EndTurn":
            picked = args[1]
            self.update_known(player.name,picked,add=True)
        # In this case, update the players public cards, based on what the player played
        # Remove played cards, and add unknown cards (Card(-1,"X"))
        elif moveid in ["InitialPlay", "PlayToOther", "PlayToSelf"]:
            played = args[-1]
            self.update_known(player.name,played,add=False)
        # If the player played a card from their hand, remove the card from the known cards
        elif moveid == "PlayFallFromHand":
            played = list(args[-1].keys())
            #fell = list(args[-1].values())
            self.update_known(player.name,played,add=False)
        # If there are only 2 players left (and no deck), we know the other players cards, and can infer the other players cards reliably
        # This can also be possible for more players, but with 2 it is trivial
        if len(self.game.get_players_condition(lambda x: x.EXIT_STATUS == 0)) == 2 and player.EXIT_STATUS == 0:
            self.game.glog.info(f"Only two players left, updating known cards")
            # Get the cards that are hidden to the player
            # If there are only two players left, we know the other players cards
            hidden_cards = self.get_hidden_cards(player)
            other_player = self.game.get_players_condition(lambda x: x.EXIT_STATUS == 0 and x.name != player.name)[0]

            # The other players cards are the hidden cards + what we already know about the other player
            self.player_cards[other_player.name] = [c for c in self.player_cards[other_player.name] if c.suit != "X"] + hidden_cards

            if len(other_player.hand.cards) != len(self.player_cards[other_player.name]):
                self.game.glog.error(f"Other players actual hand and counted cards do not match on length")
                self.game.glog.error(f"Other players actual hand: {other_player.hand.cards}")
                self.game.glog.error(f"Other players counted cards: {self.player_cards[other_player.name]}")
                raise Exception(f"Other players actual hand and counted cards do not match on length")
            if not all([c in self.player_cards[other_player.name] for c in other_player.hand.cards]):
                raise Exception(f"Other players actual hand and counted cards do not match on cards")

            other_player.plog.info(f"Updated known cards: {self.player_cards[other_player.name]}")
            other_player.plog.info(f"Actual hand: {other_player.hand.cards}")
        # After updating known cards, check if the player lifted unknown cards from deck
        self.update_unknown(player.name)
        # No updates to hand when playing from deck or skipping
        return
    
    def remove_from_game(self,cards : List[Card]) -> None:
        """ Remove cards from the card monitor. Both as keys and values in the cards_fall_dict.
        This is called at the end of a turn, and when cards are fallen from hand.
        
        Called from Turns.EndTurn.clear_table with moskaGame.fell_cards IF all cards were not lifted
        """
        # Remove the removed cards from the cards_fall_dict
        for card in cards:
            # Remove the fallen card as a key
            if card in self.cards_fall_dict:
                self.cards_fall_dict.pop(card)
        # Remove the card as value from the list
        for card_d, falls in self.cards_fall_dict.copy().items():
            for card in cards:
                if card in falls:
                    self.cards_fall_dict[card_d].remove(card)
        return
        
    
    def update_unknown(self, player_name : str) -> None:
        """Update missing cards from the players hand. Either add or remove unknown cards (Card(-1,"X"))
        """
        # The cards we know the player has
        known_cards = self.player_cards[player_name]
        # Get the cards the player actually has
        actual_cards = self.game.get_players_condition(cond = lambda x : x.name == player_name)[0].hand.cards
        # If a player has the trump card, and we do not know it yet
        # This requires, that the deck is empty. Otherwise it is just a mock move
        if any(card == self.game.trump_card for card in actual_cards) and self.game.trump_card not in known_cards and len(self.game.deck) == 0:
            self.update_known(player_name,[self.game.trump_card],add=True)
            known_cards = self.player_cards[player_name]
        missing = len(actual_cards) - len(known_cards)
        add = True
        if missing < 0:
            add = False
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
                try:
                    self.player_cards[player_name].remove(card)
                except:
                    print(f"CardMonitor: Tried to remove {card} from {player_name}, but it was not in the players hand")
                    raise ValueError(f"CardMonitor: Tried to remove {card} from {player_name}, but it was not in the players hand")
        # If we want to add cards to the players hand
        else:
            self.player_cards[player_name] = cards + self.player_cards[player_name]
        return