from __future__ import annotations
from abc import abstractmethod
from collections import Counter, namedtuple
from dataclasses import dataclass
import itertools
import logging
import random
import threading
import time
import warnings
import numpy as np
from .AbstractPlayer import AbstractPlayer
from typing import Any, Dict, List,TYPE_CHECKING, Tuple

from .utils import Assignment, _get_single_assignments, _get_assignments

from ..Game.GameState import FullGameState, GameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class AbstractEvaluatorBot(AbstractPlayer):
    """ This class is an abstract class for bots that evaluate the game states.
    This class handles finding the possible moves and generating the resulting game states.
    
    NOTE: This class has extra information, when the next immediate future state is not known!!
    This is to reduce complexity (otherwise the next immediate next states must be sampled and averaged).

    The 'evaluate_states' method must be implemented by the child class.
    
    """
    def __init__(self, moskaGame: MoskaGame = None,
                 name: str = "",
                 delay=0,
                 requires_graphic: bool = False, 
                 log_level=logging.INFO,
                 log_file="",
                 max_num_states : int = 1000,
                 ):
        # This is purely used for game complexity analysis from log files.
        self.get_nmoves = True
        self.max_num_states = max_num_states
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)
    
    @abstractmethod
    def evaluate_states(self, states : List[FullGameState]) -> List[float]:
        pass
    
    def _get_assignments(self) -> set[Assignment]:
        """ Return a set of found Assignments, containing all possible assignments of cards from the hand to the cards to fall.
        Symmetrical assignments are considered the same: where the same cards are played to the same cards, regardless of order.
        
        The assignments (partial matchings) are searched for recursively, in a depth-first manner:
        - Find all single card assignments (row-col pairs where the intersection == 1)
        - Add the assignment to found_assignments
        - Mark the vertices (row and column of the cost_matrix) as visited (0) (played_card = column, hand_card = row)
        - Repeat
        """
        return _get_assignments(from_ = self.hand.cards, to = self.moskaGame.cards_to_fall, triumph=self.moskaGame.triumph, max_num=self.max_num_states)
    
    def _get_move_prediction(self, move : str,get_n : bool = False) -> Tuple[Any,float]:
        """ Get a prediction for a moves best 'goodness' """
        # Get the possible next states for a move
        # The states should not be empty, because the next states are only computed if the move is valid
        plays, states, evals = self.get_possible_next_states(move)
        if len(plays) == 0:
            raise ValueError("No possible next states for move: ", move)
        if move == "PlayFallFromDeck":
            # Store the scores, to be able to get the best pre-computed score for a specific play
            self.play_fall_from_deck_scores = {tuple(play) : eval for play, eval in zip(plays, evals)}
            plays = ["unknown"]
            states = ["unknown"]
            evals = [float(np.mean(evals))]
        combined = list(zip(plays, states, evals))
        try:
            best = max(combined, key=lambda x : x[2])
        except:
            print("Combined: ", combined, flush=True)
            print("Evals: ", evals, flush=True)
            print("Plays: ", plays, flush=True)
            raise Exception("Could not find best play")
        #print("Best play: ", best[0], " with eval: ", best[2], flush=True)
        if get_n:
            return best[0],best[2],len(plays)
        return best[0],best[2]

    def _make_mock_move(self,move,args):
        state = self.moskaGame._make_mock_move(move,args)
        return state
    
    def _get_skip_play_states(self):
        """ Get the states for the skip play """
        plays = [[]]
        # The state should be the same, so this is likely unnecessary
        states = self._make_mock_move("Skip",[self])
        if isinstance(states,list):
            if len(states) != 1:
                raise ValueError("Expected only one state for Skip play")
            states = states[0]
        return plays, [states]
    
    def _get_play_fall_from_hand_play_states(self) -> Tuple[List[dict[Card,Card]], List[FullGameState]]:
        """ Get N possible plays and the resulting states for falling a card on the table from hand
        """
        # Get a list of tuples, where each odd index (1,3,..) is a card from hand, and each even index (0,2,..) is a card on the table
        # Ex: (hand_card1, table_card1, hand_card2, table_card2)
        # This does a DFS, and returns N possible plays
        assignments = self._get_assignments()
        self.plog.debug("Found {} possible plays".format(assignments))
        
        # Get a random sample of the plays. Evaluating each could take a long time
        # TODO: Prioritize by length
        assignments = random.sample(assignments, min(len(assignments), self.max_num_states))
        
        plays = []
        for play in assignments:
            hand_cards = [self.hand.cards[i] for i in play._hand_inds]
            table_cards = [self.moskaGame.cards_to_fall[i] for i in play._table_inds]
            plays.append({hc : tc for hc,tc in zip(hand_cards,table_cards)})
        states = []
        self.plog.debug(f"Found SAMPLE of plays to 'PlayFallFromHand': {plays}")
        for play in plays:
            # Get the state after playing 'play' from hand
            state = self._make_mock_move("PlayFallFromHand",[self, play])
            if isinstance(state,list):
                if len(state) != 1:
                    raise ValueError("Expected only one state for PlayFallFromHand")
                state = state[0]
            states.append(state)
        return plays, states
    
    def _get_play_to_self_play_states(self) -> Tuple(List[List[Card]], List[FullGameState]):
        """
        Get N possible plays and the resulting states for playing a card to self.
        This is done by finding all combinations of cards in hand that can be played to self.
        """
        playable_from_hand = self._playable_values_from_hand()
        chand = self.hand.copy()
        playable_cards = chand.pop_cards(cond=lambda c : c.value in playable_from_hand)
        self.plog.debug(f"Found playable cards to 'PlayToSelf': {playable_cards}")
        plays = []
        for i in range(1,len(playable_cards)+1):
            plays += list(itertools.combinations(playable_cards,i,))
        self.plog.debug(f"Found {len(plays)} plays to 'PlayToSelf'. Sampling {min(len(plays),self.max_num_states)}.")
        plays = random.sample(plays,min(len(plays),self.max_num_states))
        states = []
        for i,play in enumerate(plays):
            # Convert play to a list, required by Turns
            plays[i] = list(play)
            state = self._make_mock_move("PlayToSelf",[self, self, plays[i]])
            if isinstance(state,list):
                if len(state) != 1:
                    raise ValueError("Expected only one state for PlayToSelf")
                state = state[0]
            states.append(state)
        return plays, states
        
    def _get_play_from_deck_play_states(self) -> Tuple[List[Card], List[FullGameState]]:
        """ Returns a list of plays and states, that are possible from the current deck.
        If the length of a play is 2, the first card is the card to play from the deck, and the second is the card to fall.
        If the length of a play is 1, the card is the card to play to self.

        NOTE: This is a special case wrt to hidden information. Even this agent doesn't know the card from deck
        """
        cards_possibly_in_deck = self.moskaGame.card_monitor.get_cards_possibly_in_deck(self)
        plays = []
        states = []
        # Loop through all cards possibly in deck. Max about 45
        for card in cards_possibly_in_deck:
            # If the card can fall cards, make a cost matrix and get the assignments
            if self._map_to_list(card):
                assignments = _get_assignments(from_ = [card], to = self.moskaGame.cards_to_fall,triumph=self.moskaGame.triumph)
                #cm = self._make_cost_matrix([card], self.moskaGame.cards_to_fall, scoring=lambda c1,c2 : 1, max_val=0)
                #assignments = self._get_assignments(matrix=cm)
                # Evaluate the assignments. If the deck_card can fall a card, we can check the state as 'PlayFallFromHand' -play
                for assign in assignments:
                    play = [card, self.moskaGame.cards_to_fall[assign._table_inds[0]]]
                    plays.append(play)
                    # Add the card to the hand and check the state after playing the card
                    self.hand.add([card])
                    # The card must be added to hand to be able to check the state
                    self.moskaGame.card_monitor.update_unknown(self.name)
                    state = self._make_mock_move("PlayFallFromHand",[self, {play[0]:play[1]}])
                    if isinstance(state,list):
                        if len(state) != 1:
                            raise ValueError("Expected only one state for PlayFallFromHand")
                        state = state[0]
                    # Remove the card from the hand
                    self.hand.pop_cards(cond=lambda c : c == card)
                    # Remove the card and update card_monitor
                    self.moskaGame.card_monitor.update_unknown(self.name)
                    states.append(state)
            else:
                play = [card]
                plays.append(play)
                # Add the card to the hand and check the state after playing the card TO SELF
                self.hand.add(play)
                state = self._make_mock_move("PlayToSelfFromDeck",[self, self, play])
                if isinstance(state,list):
                    if len(state) != 1:
                        raise ValueError("Expected only one state for PlayToSelfFromDeck")
                    state = state[0]
                # Remove the card from the hand
                self.hand.pop_cards(cond=lambda c : c == card)
                states.append(state)
        self.plog.debug(f"{len([p for p in plays if len(p) == 2])} plays to 'PlayFallFromHand' and {len([p for p in plays if len(p) == 1])} plays to 'PlayToSelfFromDeck'.")
        return plays, states
    
    def _get_play_to_other_play_states(self) -> Tuple[List[List[Card]], List[FullGameState]]:
        """ Get N possible plays and the resulting states for playing a card to other.
        """
        playable_from_hand = self._playable_values_from_hand()
        chand = self.hand.copy()
        playable_cards = chand.pop_cards(cond=lambda c : c.value in playable_from_hand)
        self.plog.debug(f"Playable cards: {playable_cards}")
        #start = time.time()
        play_iterables = []
        for i in range(1,min(len(playable_cards)+1,self._fits_to_table()+1)):
            play_iterables.append(itertools.combinations(playable_cards,i))
        plays = list(itertools.chain.from_iterable(play_iterables))
        #self.plog.debug(f"Found plays to 'PlayToOther': {plays}")
        self.plog.debug(f"Found {len(plays)} plays to 'PlayToOther'. Sampling {min(len(plays),self.max_num_states)}.")
        plays = random.sample(plays,min(len(plays),self.max_num_states))
        states = []
        target = self.moskaGame.get_target_player()
        for i,play in enumerate(plays):
            plays[i] = list(play)
            state = self._make_mock_move("PlayToOther",[self, target, plays[i]])
            if isinstance(state,list):
                for s in state:
                    states.append(s)
            else:
                states.append(state)
        return plays, states
    
    def _get_initial_play_play_states(self) -> Tuple[List[List[Card]], List[FullGameState]]:
        """ Get N possible plays and the resulting states for playing cards to other on an Initiating turn.
        """
        cards = self.hand.copy().cards
        fits = min(self._fits_to_table(), len(cards))
        self.plog.debug(f"{fits} fits to table")
        #plays = itertools.chain.from_iterable((itertools.combinations(cards,i) for i in range(1,fits + 1)))
        play_iterables = []
        counter = Counter([c.value for c in cards])
        # We can play each single card, and all cards that have at least 2 of the same value
        for i in range(1,fits + 1):
            tmp_cards = cards.copy()
            # Filter out cards that have less than 2 of the same value
            if i > 1:
                tmp_cards = [c for c in tmp_cards if counter[c.value] >= 2]
            # Add the i length combinations to the play_iterables
            play_iterables.append(itertools.combinations(tmp_cards,i))
        # TODO: convert to a filter expression, or a generator
        plays = itertools.chain.from_iterable(play_iterables)
        legal_plays = []
        count = 0
        # TODO: See if checking legality is necessary, or is the algorithm correct.
        for play in plays:
            count += 1
            c = Counter([c.value for c in play])
            if (len(play) == 1 or all((count >= 2 for count in c.values()))):
                legal_plays.append(list(play))
        self.plog.debug(f"Tried {count} plays")
        self.plog.debug(f"Found legal plays to 'InitialPlay': {legal_plays}")
        self.plog.info(f"Found {len(legal_plays)} legal plays to 'InitialPlay'.")
        target = self.moskaGame.get_target_player()
        random.shuffle(legal_plays)
        states = []
        for i, play in enumerate(legal_plays):
            if len(states) >= self.max_num_states:
                break
            state = self._make_mock_move("InitialPlay",[self, target, list(play)])
            if isinstance(state,list):
                for s in state:
                    states.append(s)
            else:
                states.append(state)
        return legal_plays, states
    
    def get_possible_next_states(self, move : str) -> Tuple[List[Any], List[FullGameState], List[float]]:
        """ Returns a tuple containing the possible next moves,
        the corresponding states and the evaluation of the game after playing the move.
        """
        state = FullGameState.from_game(self.moskaGame,copy=True)
        self.plog.info("Getting possible next states for move: " + move)
        start = time.time()
        if move == "Skip":
            plays, states = self._get_skip_play_states()
            
        elif move == "PlayFallFromHand":
            plays, states = self._get_play_fall_from_hand_play_states()
                
        elif move == "PlayToSelf":
            plays, states = self._get_play_to_self_play_states()
                
        elif move == "PlayToOther":
            plays, states = self._get_play_to_other_play_states()
                
        elif move == "EndTurn":
            plays = [self.moskaGame.cards_to_fall.copy(), self.moskaGame.cards_to_fall.copy() + self.moskaGame.fell_cards.copy()]
            states = []
            for play in plays:
                state_ = self._make_mock_move(move,[self, play])
                if isinstance(state_,list):
                    for s in state_:
                        states.append(s)
                else:
                    states.append(state_)
        
        elif move == "InitialPlay":
            plays,states = self._get_initial_play_play_states()

        elif move == "PlayFallFromDeck":
            # NOTE: This is a special case, where the card from the deck is not known.
            plays, states = self._get_play_from_deck_play_states()
        else:
            raise Exception("Unknown move: " + move)
        self.plog.info(f"Found {len(states)} states for move {move}. Time taken: {time.time() - start}")
        is_eq, msg = state.is_game_equal(self.moskaGame,return_msg=True)
        if not is_eq:
            raise Exception("State changed during get_possible_next_states:\n" + msg)
        # TODO: Perhaps add a check for duplicate states
        start = time.time()
        predictions = self.evaluate_states(states)
        self.plog.debug(f"Time taken to evaluate {len(states)} states: {time.time() - start}")
        return plays, states, predictions
    
    
    def choose_move(self, playable: List[str]) -> str:
        self.plog.info("Choosing move...")
        self.plog.debug(f"Triumph: {self.moskaGame.triumph_card}")
        self.plog.debug(f"Hand: {self.hand}")
        self.plog.debug(f"Table: {self.moskaGame.cards_to_fall}")
        self.plog.debug(f"Fell: {self.moskaGame.fell_cards}")
        move_scores = {}
        total_n_moves = 0
        for move in playable:
            if self.get_nmoves:
                play,eval_,nmoves = self._get_move_prediction(move,get_n=True)
                move_scores[move] = (play,eval_)
                total_n_moves += nmoves
            else:
                move_scores[move] = self._get_move_prediction(move)
        if self.get_nmoves:
            self.plog.info(f"NMOVES: {len(self.moskaGame.deck)} , {total_n_moves}")
        self.plog.info(f"Move scores: {move_scores}")
        self.move_play_scores = move_scores
        best_move = max(move_scores.items(),key=lambda x: x[1][1])
        self.plog.info(f"Playing best move: {best_move}")
        return best_move[0]
    
    def play_fall_card_from_hand(self) -> Dict[Card, Card]:
        """Select random card-in-hand : card-on-table pairs

        Returns:
            Dict[Card, Card]: _description_
        """
        out = self.move_play_scores["PlayFallFromHand"][0]
        return out
    
    def play_initial(self) -> List[Card]:
        """Play a random single card, to the table on an initialization

        Returns:
            List[Card]: _description_
        """
        
        out = self.move_play_scores["InitialPlay"][0]
        return out
    
    def play_to_self(self) -> List[Card]:
        out = self.move_play_scores["PlayToSelf"][0]
        return out
    
    def play_to_target(self) -> List[Card]:
        out = self.move_play_scores["PlayToOther"][0]
        return out
    
    def deck_lift_fall_method(self, deck_card: Card) -> Tuple[Card, Card]:
        """ This a special case, where the card from the deck is not known.
        The evaluated cases are stored in self.play_fall_from_deck_scores, and the best one is returned, IF the deck card can fall a card
        """
        for play, eval in self.play_fall_from_deck_scores.items():
            if len(play) == 2 and play[0] == deck_card:
                return play
    
    def end_turn(self) -> List[Card]:
        out = self.move_play_scores["EndTurn"][0]
        return out
    
    
                
        