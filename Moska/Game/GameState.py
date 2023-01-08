import copy
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING
from .Deck import Card
if TYPE_CHECKING:
    from .Game import MoskaGame
    from Moska.Player.AbstractPlayer import AbstractPlayer
from .Deck import StandardDeck
REFERENCE_DECK = tuple(StandardDeck(shuffle = False).cards)


class GameState:
    """ A class representing a state of a MoskaGame, from the perspective of a player. Contains information about the state of the game, and methods to handle the information. """
    def __init__(self, deck_left : int, player_cards : List[List[Card]], cards_fall : Dict[Card,List[Card]], cards_on_table : List[Card], fell_cards : List[Card], player_status : List[int]):
        """Initializes a GameState object. The object contains information about the state of the game, and methods to handle the information.

        Args:
            deck_left (int): How many cards are left in the deck.
            player_cards (List[List[Card]]): A list of lists of cards. The lists are ordered by the players pid.
            cards_fall (Dict[Card,List[Card]]): A dictionary containing each card left in the game, and the cards that the card can fall.
            cards_on_table (List[Card]): A list of cards that are currently on the table.
            player_status (List[int]): A list of integers representing the status of each player. The list is ordered by the players pid. 
            - 0 : the player is not in the game.
            - 1 : the player is in the game.
            - 2 : The player has the turn to fall cards
        """
        self.deck_left = deck_left
        self.player_cards = player_cards#tuple((tuple(cards) for cards in player_cards))
        self.cards_fall = cards_fall#{card : len(cards) for card,cards in cards_fall.items()}
        self.cards_on_table = cards_on_table#tuple(cards_on_table)
        self.fell_cards = fell_cards#tuple(fell_cards)
        self.player_status = tuple(player_status)
    
    @classmethod
    def from_game(cls, game : 'MoskaGame'):
        """ Creates a GameState object from a MoskaGame object."""
        player_hands_dict = copy.deepcopy(game.card_monitor.player_cards)
        player_names = [pl.name for pl in game.players]
        player_hands = []
        # Loop through the list by pid, and add the player's hand to the list.
        for pl in player_names:
            player_hands.append(player_hands_dict[pl])
        return cls(len(game.deck), player_hands, copy.deepcopy(game.card_monitor.cards_fall_dict), game.cards_to_fall.copy(), game.fell_cards.copy(), cls._get_player_status(cls,game))
    
    def encode_cards(self, cards : List[Card],normalize : bool = False) -> List[int]:
        """Encodes a list of cards into a list of integers.
        Returns a list of 52 zeros, with the index of the card in the reference deck set to the number of cards that the card can fall.
        If a card is not in the game (not in cards_fall), the value is -1.
        """
        out = [0] * len(REFERENCE_DECK)
        if len(out) != 52:
            raise ValueError("The reference deck is not 52 cards long.")
        for card in cards:
            # If no such card exists, the value is -1, which indicates that the card is not in the game.
            #encoding_value = self.cards_fall.get(card,-1)
            encoding_value = -1
            if card in self.cards_fall:
                encoding_value = len(self.cards_fall[card])
            try:
                out[REFERENCE_DECK.index(card)] = encoding_value/52 if normalize else encoding_value
            # If the card is not in the reference deck it is likely an unknown card
            except ValueError:
                pass
        return out
    
    def as_vector(self,normalize : bool = True):
        """Returns a numeric vector representation of the state.
        The vector contains hot-encoded information about the state. The vectors are ordered by reference deck or by pid.
        The vector is ordered as follows:
        - The number of cards left in the deck.
        - The cards in each player's hand, that are known by everyone, ordered by pid.
        - All cards, and how many cards they can fall, or -1 if the card is not in the game.
        - The cards on the table, waiting to be fell
        - The cards that have fallen during this turn
        - The status of each player, ordered by pid.
        """
        player_hands = []
        player_cards = []
        for cards in self.player_cards:
            player_hands += self.encode_cards(cards)
            player_cards.append(len(cards))
        out = [self.deck_left] + player_hands + self.encode_cards(REFERENCE_DECK) + self.encode_cards(self.cards_on_table) + self.encode_cards(self.fell_cards) + player_cards
        if normalize:
            out = [x / 52 for x in out]
        out += list(self.player_status)
        # len out should be 7x52 + 1 + 4 + 4 (+52 + 1) = 426  
        return out
    
    def _get_player_status(cls, game : 'MoskaGame') -> List[int]:
        """Get the status vector of the players in the game instance.
            - 0 : the player is not in the game.
            - 1 : the player is in the game.
            - 2 : The player has the turn to fall cards
        """
        players = game.players
        statuses = []
        for pl in players:
            out = 0
            #if pl.rank is None:
            #    out += 1
            if pl is game.get_target_player():
                out += 1
            statuses.append(out)
        return tuple(statuses)
    
        