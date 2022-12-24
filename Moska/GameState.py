from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING
from Moska.Deck import Card
if TYPE_CHECKING:
    from Moska.Game import MoskaGame
    from Moska.Player.AbstractPlayer import AbstractPlayer
from Moska.Deck import StandardDeck
REFERENCE_DECK = list(StandardDeck(shuffle = False).cards)


class GameState:
    """ A class representing a state of a MoskaGame, from the perspective of a player. Contains information about the state of the game, and methods to handle the information. """
    def __init__(self, deck_left : int, player_cards : List[List[Card]], cards_fall : Dict[Card,List[Card]], cards_on_table : List[Card], player_status : List[int]):
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
        self.player_cards = tuple((tuple(cards) for cards in player_cards))
        self.cards_fall = {card : len(cards) for card,cards in cards_fall.items()}
        self.cards_on_table = tuple(cards_on_table)
        self.player_status = tuple(player_status)
    
    @classmethod
    def from_game(cls, game : 'MoskaGame'):
        """ Creates a GameState object from a MoskaGame object."""
        player_hands_dict = game.card_monitor.player_cards
        player_names = [pl.name for pl in game.players]
        player_hands = []
        # Loop through the list by pid, and add the player's hand to the list.
        for pl in player_names:
            player_hands.append(player_hands_dict[pl])
        return cls(len(game.deck), player_hands, game.card_monitor.cards_fall_dict.copy(), game.cards_to_fall.copy(), cls._get_player_status(cls,game))
    
    def encode_cards(self, cards : List[Card]) -> List[int]:
        """Encodes a list of cards into a list of integers.
        Returns a list of 52 zeros, with the index of the card in the reference deck set to the number of cards that the card can fall.
        If a card is not in the game (not in cards_fall), the value is -1.
        """
        out = [0] * len(REFERENCE_DECK)
        if len(out) != 52:
            raise ValueError("The reference deck is not 52 cards long.")
        for card in cards:
            # If no such card exists, the value is -1, which indicates that the card is not in the game.
            encoding_value = self.cards_fall.get(card,-1)
            try:
                out[REFERENCE_DECK.index(card)] = encoding_value
            # If the card is not in the reference deck it is likely an unknown card
            except ValueError:
                pass
        return out
    
    def as_vector(self):
        """Returns a numeric vector representation of the state."""
        player_hands = []
        for cards in self.player_cards:
            player_hands += self.encode_cards(cards)
        out = [self.deck_left] + player_hands + self.encode_cards(list(self.cards_fall.keys())) + self.encode_cards(self.cards_on_table) + list(self.player_status)
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
            if pl.rank is None:
                out += 1
            if pl is game.get_target_player():
                out += 1
            statuses.append(out)
        return tuple(statuses)
    
        