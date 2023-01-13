import copy
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING
import warnings
from .Deck import Card
if TYPE_CHECKING:
    from .Game import MoskaGame
    from Moska.Player.AbstractPlayer import AbstractPlayer
from .Deck import StandardDeck
REFERENCE_DECK = tuple(StandardDeck(shuffle = False).cards)


class FullGameState:
    def __init__(self,*args, copy : bool = False):
        self.copied = copy
        if copy:
            self.__init__with_copy(*args)
        else:
            self.__init__no_copy(*args)
    
    def __init__no_copy(self,
                 deck : StandardDeck,
                 known_player_cards : List[Card],
                 full_player_cards : List[Card],
                 fell_cards : List[Card],
                 cards_to_fall : List[Card],
                 cards_fall_dict : Dict[Card,List[Card]],
                 players_ready : List[bool],
                 players_in_game : List[bool],
                 tc_index : int,
                 ):
        self.deck = deck # The deck of cards
        self.full_player_cards = full_player_cards # Complete information about the players current cards
        self.known_player_cards = known_player_cards # Known player cards
        self.fell_cards = fell_cards # Cards that have fallen and are on the table
        self.cards_to_fall = cards_to_fall # Cards played to the target, that have not yet fallen
        self.cards_fall_dict = cards_fall_dict # A dictionary containing each card left in the game, and the cards that the card can fall.
        self.players_ready = players_ready # A list of booleans representing if each player is ready
        self.players_in_game = players_in_game # A list of booleans representing if each player is in the game
        self.tc_index = tc_index # The index in turn cycle. Ensures the target player is saved
        
        
    def __init__with_copy(self,
                deck : StandardDeck,
                known_player_cards : List[Card],
                full_player_cards : List[Card],
                fell_cards : List[Card],
                cards_to_fall : List[Card],
                cards_fall_dict : Dict[Card,List[Card]],
                players_ready : List[bool],
                players_in_game : List[bool],
                tc_index : int,
                ):
        self.deck = copy.deepcopy(deck) # The deck of cards
        self.full_player_cards = copy.deepcopy(full_player_cards) # Complete information about the players current cards
        self.known_player_cards = copy.deepcopy(known_player_cards) # Known player cards
        self.fell_cards = copy.deepcopy(fell_cards) # Cards that have fallen and are on the table
        self.cards_to_fall = copy.deepcopy(cards_to_fall) # Cards played to the target, that have not yet fallen
        self.cards_fall_dict = copy.deepcopy(cards_fall_dict) # A dictionary containing each card left in the game, and the cards that the card can fall.
        self.players_ready = copy.copy(players_ready) # A list of booleans representing if each player is ready
        self.players_in_game = copy.copy(players_in_game) # A list of booleans representing if each player is in the game
        self.tc_index = tc_index # The index in turn cycle. Ensures the target player is saved
        
        
    def restore_game_state(self,game : 'MoskaGame', check : bool = False):
        if check:
            passed, msg = self.is_game_equal(game,return_msg=True)
            if not passed:
                raise ValueError(msg)
        if not self.copied:
            warnings.warn("The game state was not copied. This may cause unexpected behaviour.")
        # These should be fine
        game.deck = self.deck
        game.fell_cards = self.fell_cards
        game.cards_to_fall = self.cards_to_fall
        game.card_monitor.cards_fall_dict = self.cards_fall_dict
        game.card_monitor.player_cards = {pl.name:cards for pl,cards in zip(game.players,self.known_player_cards)}
        game.turnCycle.ptr = self.tc_index
        
        for pl in game.players:
            pl.ready = self.players_ready[pl.pid]
            # The players RANKS can not be restored but if player states are fine, so should the ranks.
            # Restoring incorrect ranks would require starting
            # the players again, and that is not currently possible.
        for pl, cards in zip(game.players,self.full_player_cards):
            pl.hand.cards = cards
        return
    
    @classmethod
    def from_game(cls,game : 'MoskaGame', copy : bool = True):
        full_player_cards = [pl.hand.cards for pl in game.players]
        known_player_cards = [game.card_monitor.player_cards[pl.name] for pl in game.players]
        players_ready = [pl.ready for pl in game.players]
        players_in_game = [pl.rank is None for pl in game.players]
        return cls(game.deck,
                   known_player_cards,
                   full_player_cards,
                   game.fell_cards,
                   game.cards_to_fall,
                   game.card_monitor.cards_fall_dict,
                   players_ready,
                   players_in_game,
                   game.turnCycle.ptr,
                   copy = copy,
                   )
        
    def is_game_equal(self, other : 'MoskaGame', return_msg : bool = False):
        out = True
        msg = ""
        if self.tc_index != other.turnCycle.ptr:
            out = False
            msg = "The tc_index is not equal. {} != {}".format(self.tc_index,other.turnCycle.ptr)
        elif [card.kopled for card in self.cards_to_fall] != [card.kopled for card in other.cards_to_fall]:
            out = False
            msg = "The cards_to_fall kopled state is not equal. {} != {}".format(self.cards_to_fall,other.cards_to_fall)
        elif self.deck.cards != other.deck.cards:
            out = False
            msg = "The decks are not equal. {} != {}".format(self.deck.cards,other.deck.cards)
        elif self.full_player_cards != [pl.hand.cards for pl in other.players]:
            out = False
            msg = "The full player cards are not equal. {} != {}".format(self.full_player_cards,[pl.hand.cards for pl in other.players])
        
        elif self.known_player_cards != [other.card_monitor.player_cards[pl.name] for pl in other.players]:
            out = False
            msg = "The known player cards are not equal. {} != {}".format(self.known_player_cards,[other.card_monitor.player_cards[pl.name] for pl in other.players])
        elif self.fell_cards != other.fell_cards:
            out = False
            msg = "The fell cards are not equal. {} != {}".format(self.fell_cards,other.fell_cards)
        elif self.cards_to_fall != other.cards_to_fall:
            out = False
            msg = "The cards to fall are not equal. {} != {}".format(self.cards_to_fall,other.cards_to_fall)
        elif self.cards_fall_dict != other.card_monitor.cards_fall_dict:
            out = False
            msg = "The cards fall dict are not equal. {} != {}".format(self.cards_fall_dict,other.card_monitor.cards_fall_dict)
        elif self.players_ready != [pl.ready for pl in other.players]:
            out = False
            msg = "The players ready are not equal. {} != {}".format(self.players_ready,[pl.ready for pl in other.players])
        elif self.players_in_game != [pl.rank is None for pl in other.players]:
            out = False
            msg = "The players in game are not equal. {} != {}".format(self.players_in_game,[pl.rank is None for pl in other.players])
        if return_msg:
            return out,msg
        return out

    def encode_cards(self, cards : List[Card],cards_fall_dict = None) -> List[int]:
        """Encodes a list of cards into a list of integers.
        Returns a list of 52 zeros, with the index of the card in the reference deck set to the number of cards that the card can fall.
        If a card is not in the game (not in cards_fall), the value is -1.
        """
        if not cards_fall_dict:
            cards_fall_dict = self.cards_fall_dict
        out = [0] * len(REFERENCE_DECK)
        if len(out) != 52:
            raise ValueError("The reference deck is not 52 cards long.")
        # Loop through input cards
        for card in cards:
            # If the card is an uknown, skip it.
            if card == Card(-1,"X"):
                continue
            # If no such card exists, the value is -1, which indicates that the card is not in the game.
            encoding_value = -1
            if card in cards_fall_dict:
                encoding_value = len(cards_fall_dict[card])
            # The index of the card in the reference deck is set to the number of
            # cards that the card can fall OR -1 if the card is not in the game.
            out[REFERENCE_DECK.index(card)] = encoding_value
        return out
    
    def as_full_information_vector(self):
        out = []
        # How many cards are left in the deck
        out += [len(self.deck.cards)]
        # How many cards each player has in their hand
        out += [len(hand) for hand in self.known_player_cards]
        # Which cards are still in the game, and encoded as how many cards they can fall.
        out += self.encode_cards(REFERENCE_DECK)
        # Which cards are on the table, waiting to be fell
        out += self.encode_cards(self.cards_to_fall)
        # Which cards have fallen during this turn
        out += self.encode_cards(self.fell_cards)
        # Whether each player is ready, ordered by pid. This tells whether the player might play new cards to the current table.
        out += [1 if ready else 0 for ready in self.players_ready]
        # Whether each player is in the game, ordered by pid. This tells whether the player is still in the game.
        out += [1 if in_game else 0 for in_game in self.players_in_game]
        # Whether there is kopled card on the table
        out += 1 if any([card.kopled for card in self.cards_to_fall]) else 0
        # Add a vector of the order of the cards in the deck, encoded as the number of cards that the card can fall.
        deck_order = [0 for _ in range(52)]
        for i,card in enumerate(self.deck.cards):
            deck_order[i] = self.cards_fall_dict[card]
        out += deck_order
        
        # Full information about the cards in each player's hand
        for cards in self.full_player_cards:
            out += self.encode_cards(cards)
        return out
        
    def as_perspective_vector(self, player : 'AbstractPlayer'):
        """Returns a numeric vector representation of the state.
        The vector contains hot-encoded information about the state. The vectors are ordered by reference deck or by pid.
        The vector is ordered as follows:
        - The number of cards in the deck.
        - The number of cards in each player's hand
        - All cards, and how many cards they can fall, or -1 if the card is not in the game anymore.
        - The cards on the table, waiting to be fell
        - The cards that have fallen during this turn
        - The status of each player; whether they are ready to play new cards to the table.
        - The status of each player; whether they are still in the game.
        - Whether there is a kopled card on the table.
        - Cards in each players hand, known to everyone. Recorded by card monitor
        """
        out = []
        # How many cards are left in the deck
        out += [len(self.deck.cards)]
        # How many cards each player has in their hand
        out += [len(hand) for hand in self.known_player_cards]
        # Which cards are still in the game, and encoded as how many cards they can fall.
        out += self.encode_cards(REFERENCE_DECK)
        # Which cards are on the table, waiting to be fell
        out += self.encode_cards(self.cards_to_fall)
        # Which cards have fallen during this turn
        out += self.encode_cards(self.fell_cards)
        # Whether each player is ready, ordered by pid. This tells whether the player might play new cards to the current table.
        out += [1 if ready else 0 for ready in self.players_ready]
        # Whether each player is in the game, ordered by pid. This tells whether the player is still in the game.
        out += [1 if in_game else 0 for in_game in self.players_in_game]
        # Whether there is kopled card on the table
        out += [1] if any([card.kopled for card in self.cards_to_fall]) else [0]
        # Encoded player hands from the perspective of the player: All picked up cards
        for known_cards in self.known_player_cards:
            out += self.encode_cards(known_cards)
        # Encode the players own hand (full information)
        out += self.encode_cards(player.hand.cards)
        # len should be 1 + 4 +52 + 52 + 52 + 4 + 4 + 1 + 4*52 (+1) = 431/432
        return out


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
        """
        self.deck_left = deck_left
        self.player_cards = player_cards
        self.cards_fall = cards_fall
        self.cards_on_table = cards_on_table
        self.fell_cards = fell_cards
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
            #encoding_value = self.cards_fall.get(card,-1)
            encoding_value = -1
            if card in self.cards_fall:
                encoding_value = len(self.cards_fall[card])
            try:
                out[REFERENCE_DECK.index(card)] = encoding_value
            # If the card is not in the reference deck it is likely an unknown card
            except ValueError:
                pass
        return out
    
    def as_vector(self,normalize : bool = False):
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
        if normalize:
            raise DeprecationWarning("Normalizing the vector is not supported.")
        player_hands = []
        player_cards = []
        for cards in self.player_cards:
            player_hands += self.encode_cards(cards)
            player_cards.append(len(cards))
        out = [self.deck_left] + player_hands + self.encode_cards(REFERENCE_DECK) + self.encode_cards(self.cards_on_table) + self.encode_cards(self.fell_cards) + player_cards
        out += list(self.player_status)
        # len out should be 7x52 + 1 + 4 + 4 (+52 + 1) = 426  
        return out
    
    def _get_player_status(cls, game : 'MoskaGame', fmt = "new") -> List[int]:
        """Get the status vector of the players in the game instance.
            - -1 : the player is not in the game.
            - 0 : the player is in the game.
            - 1 : the player is in the game, and is not ready.
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
    
        