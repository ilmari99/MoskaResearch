import copy
from typing import Dict, List, TYPE_CHECKING, Tuple
import warnings
from .Deck import Card
if TYPE_CHECKING:
    from .Game import MoskaGame
    from Moska.Player.AbstractPlayer import AbstractPlayer
from .Deck import StandardDeck
REFERENCE_DECK = tuple(StandardDeck(shuffle = False).cards)


class FullGameState:
    """ A class representing the full game state.
    This is an entirely static representation and is not perfect, i.e. you can not restore an arbitrary games state from this.
    You atleast need the player instances.
    This used to represent the game as a vector, and to store information about the game and later to restore it (mock move).
    """
    def __init__(self,*args, copy : bool = False):
        """ Initilize the game state, with either copying everything or not. """
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
                 target_pid : int,
                 trump : str
                 ):
        """ Initilize the game state, without copying everything.
        This is faster, but if you modify the game state, this instance will be modified as well.
        """
        self.deck = deck # The deck of cards
        self.full_player_cards = full_player_cards # Complete information about the players current cards
        self.known_player_cards = known_player_cards # Known player cards
        self.fell_cards = fell_cards # Cards that have fallen and are on the table
        self.cards_to_fall = cards_to_fall # Cards played to the target, that have not yet fallen
        self.cards_fall_dict = cards_fall_dict # A dictionary containing each card left in the game, and the cards that the card can fall.
        self.players_ready = players_ready # A list of booleans representing if each player is ready
        self.players_in_game = players_in_game # A list of booleans representing if each player is in the game
        self.tc_index = tc_index # The index in turn cycle. Ensures the target player is saved
        self.target_pid = target_pid
        self.trump = trump
        
        
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
                target_pid : int,
                trump : str
                ):
        """ Initilize the game state, with copying everything.
        This is slower, but if you modify the game state, this instance will not be modified.
        All mutable objects are deepcopied.
        """
        self.deck = copy.deepcopy(deck) # The deck of cards
        self.full_player_cards = copy.deepcopy(full_player_cards) # Complete information about the players current cards
        self.known_player_cards = copy.deepcopy(known_player_cards) # Known player cards
        self.fell_cards = copy.deepcopy(fell_cards) # Cards that have fallen and are on the table
        self.cards_to_fall = copy.deepcopy(cards_to_fall) # Cards played to the target, that have not yet fallen
        self.cards_fall_dict = copy.deepcopy(cards_fall_dict) # A dictionary containing each card left in the game, and the cards that the card can fall.
        self.players_ready = copy.copy(players_ready) # A list of booleans representing if each player is ready
        self.players_in_game = copy.copy(players_in_game) # A list of booleans representing if each player is in the game
        self.tc_index = tc_index # The index in turn cycle. Ensures the target player is saved
        self.target_pid = target_pid
        self.trump = trump
        
        
    def restore_game_state(self,game : 'MoskaGame', check : bool = False) -> None:
        """ Restore the game state of a game.
        If check is True, the game state is checked to be equal to this game state after restoration.
        If check is False, the game state is not checked after restoration.
        If check is True and the game state is not equal to this game state, a ValueError is raised.
        """
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
        if check:
            passed, msg = self.is_game_equal(game,return_msg=True)
            if not passed:
                raise ValueError(msg)
        return      

    @classmethod
    def from_game(cls,game : 'MoskaGame', copy : bool = True) -> 'FullGameState':
        """ Create a game state from a game."""
        full_player_cards = [pl.hand.cards for pl in game.players]
        known_player_cards = [game.card_monitor.player_cards[pl.name] for pl in game.players]
        players_ready = [pl.ready for pl in game.players]
        players_in_game = [pl.rank is None for pl in game.players]
        target_pid = game.get_target_player().pid
        return cls(game.deck,
                   known_player_cards,
                   full_player_cards,
                   game.fell_cards,
                   game.cards_to_fall,
                   game.card_monitor.cards_fall_dict,
                   players_ready,
                   players_in_game,
                   game.turnCycle.ptr,
                   target_pid,
                   game.triumph,
                   copy = copy,
                   )

    def copy(self) -> 'FullGameState':
        """ Create a shallow copy of this object.
        This is used when sampling the possible future game states in some models.
        """
        return type(self)(self.deck,
                          [copy.copy(cards) for cards in self.known_player_cards],
                          [copy.copy(cards) for cards in self.full_player_cards],
                          copy.copy(self.fell_cards),
                          copy.copy(self.cards_to_fall),
                          copy.copy(self.cards_fall_dict),
                          copy.copy(self.players_ready),
                          copy.copy(self.players_in_game),
                          self.tc_index,
                          self.target_pid,
                          self.trump,
                          copy = False,
                          )
        
    def is_game_equal(self, other : 'MoskaGame', return_msg : bool = False) -> Tuple[bool,str]:
        """ Check if the game state is equal to the game state of a game instance.
        If return_msg is True, a tuple of a boolean and a string is returned.
        """
        out = True
        msg = ""
        # Check if the turncycle index is equal
        if self.tc_index != other.turnCycle.ptr:
            out = False
            msg = "The tc_index is not equal. {} != {}".format(self.tc_index,other.turnCycle.ptr)
        # Check if the kopled status is the same
        elif [card.kopled for card in self.cards_to_fall] != [card.kopled for card in other.cards_to_fall]:
            out = False
            msg = "The cards_to_fall kopled state is not equal. {} != {}".format(self.cards_to_fall,other.cards_to_fall)
        # Check if the cards in the deck are the same and in same order
        elif self.deck.cards != other.deck.cards:
            out = False
            msg = "The decks are not equal. {} != {}".format(self.deck.cards,other.deck.cards)
        # Check if the players have the same cards
        elif self.full_player_cards != [pl.hand.cards for pl in other.players]:
            out = False
            msg = "The full player cards are not equal. {} != {}".format(self.full_player_cards,[pl.hand.cards for pl in other.players])
        # Check if the publically known cards are the same
        elif self.known_player_cards != [other.card_monitor.player_cards[pl.name] for pl in other.players]:
            out = False
            msg = "The known player cards are not equal. {} != {}".format(self.known_player_cards,[other.card_monitor.player_cards[pl.name] for pl in other.players])
        # Check if the fell cards are the same
        elif self.fell_cards != other.fell_cards:
            out = False
            msg = "The fell cards are not equal. {} != {}".format(self.fell_cards,other.fell_cards)
        # Check if the cards to fall are the same
        elif self.cards_to_fall != other.cards_to_fall:
            out = False
            msg = "The cards to fall are not equal. {} != {}".format(self.cards_to_fall,other.cards_to_fall)
        # Check if the cards in game are the same
        elif self.cards_fall_dict != other.card_monitor.cards_fall_dict:
            out = False
            msg = "The cards fall dict are not equal. {} != {}".format(self.cards_fall_dict,other.card_monitor.cards_fall_dict)
        # Check if the players ready are the same
        elif self.players_ready != [pl.ready for pl in other.players]:
            out = False
            msg = "The players ready are not equal. {} != {}".format(self.players_ready,[pl.ready for pl in other.players])
        # Check if the players still in the game are the same
        elif self.players_in_game != [pl.rank is None for pl in other.players]:
            out = False
            msg = "The players in game are not equal. {} != {}".format(self.players_in_game,[pl.rank is None for pl in other.players])
        if return_msg:
            return out,msg
        return out
    
    def _get_card_score(self,card : Card) -> int:
        return len(self.cards_fall_dict[card])

    def encode_cards(self, cards : List[Card],fill = -1,cards_fall_dict = None) -> List[int]:
        """ DEPRECATED!!!!
        Encodes a list of cards into a list of integers.
        Returns a list of 52 zeros, with the index of the card in the reference deck set to the number of cards that the card can fall.
        If a card is not in the game (not in cards_fall), the value is -1.
        """
        warnings.warn("The card encoding method is deprecated",DeprecationWarning)
        if not cards_fall_dict:
            cards_fall_dict = self.cards_fall_dict
        out = [fill] * len(REFERENCE_DECK)
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
        raise NotImplementedError("This method is not implemented correctly yet.")
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
    
    def _as_perspective_bitmap_vector(self,player : 'AbstractPlayer') -> List[int]:
        """ Returns a vector of the game state from the perspective of the given player.
        This is the latest version of the encoding.
        TODO: Add the description of the encoding.
        """
        out = []
        out += [len(self.deck.cards)]
        out += [len(hand) for hand in self.known_player_cards]
        out += [1 if ready else 0 for ready in self.players_ready]
        out += [1 if in_game else 0 for in_game in self.players_in_game]
        out += [1 if any([card.kopled for card in self.cards_to_fall]) else 0]
        out += [1 if i == self.target_pid else 0 for i in range(4)]
        out += [1 if c.suit == self.trump else 0 for c in REFERENCE_DECK[0:4]]
        out += [1 if i == player.pid else 0 for i in range(4)]
        # Initialize a vector of zeros. Copy it, and set the index of the (in refr. deck) card to 1.
        z_init = [0 for _ in range(52)]

        z = z_init.copy()
        for card in self.cards_fall_dict:
            z[REFERENCE_DECK.index(card)] = 1
        out += z

        z = z_init.copy()
        for card in self.cards_to_fall:
            z[REFERENCE_DECK.index(card)] = 1
        out += z

        z = z_init.copy()
        for card in self.fell_cards:
            z[REFERENCE_DECK.index(card)] = 1
        out += z

        for pl, cards in enumerate(self.known_player_cards):
            z = z_init.copy()
            for card in cards:
                if card == Card(-1,"X"):
                    continue
                z[REFERENCE_DECK.index(card)] = 1
            out += z

        z = z_init.copy()
        for card in self.full_player_cards[player.pid]:
            z[REFERENCE_DECK.index(card)] = 1
        out += z
        return out
        
    def as_perspective_vector(self, player : 'AbstractPlayer', fmt : str = "new"):
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
        if fmt == "old-algbr":
            local_encode_cards = lambda cards : self.encode_cards(cards,fill=0)
        elif fmt == "new-algbr":
            local_encode_cards = self.encode_cards
        elif fmt == "bitmap":
            return self._as_perspective_bitmap_vector(player)
        else:
            raise NameError("Unknown format: {}".format(fmt))
        out = []
        # How many cards are left in the deck
        out += [len(self.deck.cards)]
        # How many cards each player has in their hand
        out += [len(hand) for hand in self.known_player_cards]
        # Which cards are still in the game, and encoded as how many cards they can fall.
        out += local_encode_cards(REFERENCE_DECK)
        # Which cards are on the table, waiting to be fell
        out += local_encode_cards(self.cards_to_fall)
        # Which cards have fallen during this turn
        out += local_encode_cards(self.fell_cards)
        # Whether each player is ready, ordered by pid. This tells whether the player might play new cards to the current table.
        out += [1 if ready else 0 for ready in self.players_ready]
        # Whether each player is in the game, ordered by pid. This tells whether the player is still in the game.
        in_game_vec = [1 if in_game else 0 for in_game in self.players_in_game]
        # In the new format, we mark player as a two, if they are in the game, and the target
        in_game_vec[self.target_pid] = 2 if in_game_vec[self.target_pid] == 1 else 0
        out += in_game_vec
        # Whether there is kopled card on the table
        out += [1] if any([card.kopled for card in self.cards_to_fall]) else [0]
        # Encoded player hands from the perspective of the player: All picked up cards
        for known_cards in self.known_player_cards:
            out += local_encode_cards(known_cards)
        # Encode the players own hand (full information)
        player_cards = self.full_player_cards[player.pid]
        out += local_encode_cards(player_cards)
        # len should be 1 + 4 +52 + 52 + 52 + 4 + 4 + 1 + 4*52 (+1) = 431/432
        return out
    
        