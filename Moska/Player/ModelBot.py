from __future__ import annotations
from collections import Counter, namedtuple
import itertools
import logging
import random
import time
import numpy as np
import copy
from .AbstractPlayer import AbstractPlayer
from typing import Dict, List,TYPE_CHECKING, Tuple
import tensorflow as tf
from ..Game.GameState import GameState
if TYPE_CHECKING:
    from ..Game.Deck import Card
    from ..Game.Game import MoskaGame

class ModelBot(AbstractPlayer):
    def __init__(self, moskaGame: MoskaGame = None, name: str = "", delay=10 ** -6, requires_graphic: bool = False, log_level=logging.INFO, log_file="", max_num_states = 600):
        if not name:
            name = "M-"
        self.max_num_states = max_num_states
        #self.model = tf.keras.models.load_model(model_file,compile=False)
        #converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        #self.model = converter.convert()
        # Load a tflite model and allocate tensors.
        self.move_play_scores = {}
        self.play_fall_from_deck_scores = {}
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)
    
    def _play_move(self) -> Tuple[bool, str]:
        out = super()._play_move()
        self.move_play_scores = {}
        return out
    
    def _get_single_assignments(self, matrix):
        """ Return all single card assignments from the matrix"""
        nz = np.nonzero(matrix)
        inds = zip(nz[0], nz[1])
        inds = [list(i) for i in inds]
        return inds
    
    
    def get_assignments(self, start=[], max_assignments : int = 100, found_assignments = None, matrix = None):
        """ Return a list of lists, containing all possible assignments of cards from the hand to the cards to fall."""
        #print("Finding assignments", flush=True)
        if not found_assignments:
            found_assignments = set()
        #print("Found assignments: ", found_assignments, flush=True)
        # If no start is given, find the first assignment
        # If no matrix is given, create the matrix
        if matrix is None:
            matrix = self._make_cost_matrix(from_ = self.hand.cards, to = self.moskaGame.cards_to_fall,max_val=0,scoring=lambda hc,tc : 1)
            
        # If we have found the maximum number of assignments, OR there are no assignments, return the found assignments
        new_assignments = self._get_single_assignments(matrix)
        
        if len(found_assignments) >= max_assignments or not new_assignments:
            return found_assignments
        
        #if not start:
        #    print("Hand: ", self.hand, flush=True)
        #    print("Cards to fall: ", self.moskaGame.cards_to_fall, flush=True)
        #    print("New assignments: ", new_assignments, flush=True)
        #print("Matrix: \n", matrix, flush=True)
        
        for row, col in new_assignments:
            found_assignments.add(tuple(start + [row,col]))
            # Get the visited cards
            # The cards in hand are even (0,2...), the cards on the table are odd (1,3...)
            hand_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 0]
            table_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 1]
            # Store the values, to restore them later
            row_vals = matrix[hand_cards,:]
            col_vals = matrix[:,table_cards]
            # Mark the cards as visited (0)
            matrix[hand_cards,:] = 0
            matrix[:,table_cards] = 0
            # Find the next assignments, which adds the new assignments to the found_assignments
            self.get_assignments(start +[row,col], max_assignments, found_assignments, matrix)
            # Restore matrix
            matrix[hand_cards,:] = row_vals
            matrix[:,table_cards] = col_vals
        return found_assignments
    
    def get_prediction(self, move : str):
        """ Get a prediction for a moves best 'goodness' """
        plays, states, evals = self.get_possible_next_states(move, num_states=self.max_num_states)
        if move == "PlayFallFromDeck":
            self.play_fall_from_deck_scores = {tuple(play) : eval for play, eval in zip(plays, evals)}
            plays = ["unknown"]
            states = ["unknown"]
            evals = [np.mean(evals)]
        combined = list(zip(plays, states, evals))
        best = max(combined, key=lambda x : x[2])
        #print("Best play: ", best[0], " with eval: ", best[2], flush=True)
        return best[0],best[2]
        
    def get_possible_next_states(self, move : str, num_states : int = 100):
        state = GameState.from_game(self.moskaGame)
        if move == "Skip":
            plays = [[]]
            states = [state.as_vector(normalize=False) + state.encode_cards(self.hand.cards)]
            
        if move == "PlayFallFromHand":
            play_indices = self.get_assignments(max_assignments=num_states)
            #print("Found plays: ", play_indices, flush=True)
            plays = []
            for play in play_indices:
                hand_cards = [self.hand.cards[ind] for i,ind in enumerate(play) if i % 2 == 0]
                table_cards = [self.moskaGame.cards_to_fall[ind] for i,ind in enumerate(play) if i % 2 == 1]
                #hand_cards = [self.hand.cards[i] for i in range(0,len(play),2)]
                #table_cards = [self.moskaGame.cards_to_fall[i] for i in range(1,len(play),2)]
                plays.append({hc : tc for hc,tc in zip(hand_cards,table_cards)})
            states = []
            self.plog.debug(f"Found plays: {plays}")
            self.plog.info(f"Found {len(plays)} plays")
            random.shuffle(plays)
            for play in plays:
                if len(states) >= num_states:
                    break
                new_state = self.moskaGame._make_mock_move(move,[self, play])
                state_vector = new_state.as_vector(normalize=False)
                chand = self.hand.copy()
                chand.pop_cards(cond=lambda c : c in play.keys())
                state_vector += state.encode_cards(chand.cards)
                #print(state_vector, flush=True)
                states.append(state_vector)
                
        if move == "PlayToSelf":
            playable_from_hand = self._playable_values_from_hand()
            chand = self.hand.copy()
            playable_cards = chand.pop_cards(cond=lambda c : c.value in playable_from_hand)
            plays = []
            for i in range(1,len(playable_cards)+1):
                plays += list(itertools.combinations(playable_cards,i))
            random.shuffle(plays)
            states = []
            for i,play in enumerate(plays):
                if len(states) >= num_states:
                    break
                plays[i] = list(play)
                new_state = self.moskaGame._make_mock_move(move,[self, self, plays[i]])
                state_vector = new_state.as_vector(normalize=False)
                chand = self.hand.copy()
                chand.pop_cards(cond=lambda c : c in play)
                state_vector += state.encode_cards(chand.cards)
                states.append(state_vector)
                
        if move == "PlayToOther":
            playable_from_hand = self._playable_values_from_hand()
            chand = self.hand.copy()
            playable_cards = chand.pop_cards(cond=lambda c : c.value in playable_from_hand)
            plays = []
            for i in range(1,min(len(playable_cards)+1,self._fits_to_table()+1)):
                plays += list(itertools.combinations(playable_cards,i))
            random.shuffle(plays)
            states = []
            target = self.moskaGame.get_target_player()
            for i,play in enumerate(plays):
                if len(states) >= num_states:
                    break
                plays[i] = list(play)
                # TODO: Currently the model has perfect information about the lifted cards
                new_state = self.moskaGame._make_mock_move(move,[self, target, plays[i]])
                state_vector = new_state.as_vector(normalize=False)
                chand = self.hand.copy()
                chand.pop_cards(cond=lambda c : c in play)
                state_vector += state.encode_cards(chand.cards)
                states.append(state_vector)
                
        if move == "EndTurn":
            plays = [self.moskaGame.cards_to_fall.copy(), self.moskaGame.cards_to_fall.copy() + self.moskaGame.fell_cards.copy()]
            states = []
            for play in plays:
                # TODO: Currently the model has perfect information about the lifted cards
                new_state = self.moskaGame._make_mock_move(move,[self, play])
                state_vector = new_state.as_vector(normalize=False) + new_state.encode_cards(self.hand.cards + play)
                states.append(state_vector)
        
        if move == "InitialPlay":
            chand = self.hand.copy()
            plays = []
            for i in range(1,min(len(chand.cards)+1, self._fits_to_table()+1)):
                plays += list(itertools.combinations(chand.cards,i))
            target = self.moskaGame.get_target_player()
            random.shuffle(plays)
            legal_plays = []
            states = []
            
            cards_possibly_in_deck = self.moskaGame.card_monitor.get_cards_possibly_in_deck(self)
            
            for i, play in enumerate(plays):
                c = Counter([c.value for c in play])
                if len(states) >= num_states:
                    break
                if not (len(play) == 1 or all((count >= 2 for count in c.values()))):
                    continue
                legal_plays.append(list(play))
                
                new_state = self.moskaGame._make_mock_move(move,[self, target, list(play)])
                
                chand = self.hand.copy()
                chand.pop_cards(cond=lambda c : c in play)
                state_vector = new_state.as_vector(normalize=False) + new_state.encode_cards(chand.cards)
                states.append(state_vector)
            plays = legal_plays
        
        if move == "PlayFallFromDeck":
            cards_possibly_in_deck = self.moskaGame.card_monitor.get_cards_possibly_in_deck(self)
            plays = []
            states = []
            # Loop through all cards possibly in deck. Max about 45
            for card in cards_possibly_in_deck:
                # If the card can fall cards, make a cost matrix and get the assignments
                if self._map_to_list(card):
                    cm = self._make_cost_matrix([card], self.moskaGame.cards_to_fall, scoring=lambda c1,c2 : 1, max_val=0)
                    assignments = self.get_assignments(matrix=cm,max_assignments=num_states)
                    # Evaluate the assignments. If the deck_card can fall a card, we can check the state as 'PlayFallFromHand' -play
                    for deck_card_i, table_card_i in assignments:
                        play = [card, self.moskaGame.cards_to_fall[table_card_i]]
                        plays.append(play)
                        
                        # Add the card to the hand and check the state after playing the card
                        self.hand.add([card])
                        #print(self.moskaGame.card_monitor.player_cards[self.name], flush=True)
                        self.moskaGame.card_monitor.update_unknown(self.name)
                        new_state = self.moskaGame._make_mock_move("PlayFallFromHand",[self, {play[0]:play[1]}])
                        # Remove the card from the hand
                        self.hand.pop_cards(cond=lambda c : c == card)
                        #print(self.hand)
                        self.moskaGame.card_monitor.update_unknown(self.name)
                        #print(self.moskaGame.card_monitor.player_cards[self.name], flush=True)
                        
                        state_vector = new_state.as_vector(normalize=False) + new_state.encode_cards(self.hand.cards)
                        states.append(state_vector)
                else:
                    play = [card]
                    plays.append(play)
                    # Add the card to the hand and check the state after playing the card TO SELF
                    self.hand.add(play)
                    new_state = self.moskaGame._make_mock_move("PlayToSelfFromDeck",[self, self, play])
                    state_vector = new_state.as_vector(normalize=False)
                    # Remove the card from the hand
                    self.hand.pop_cards(cond=lambda c : c == card)
                    
                    state_vector += state.encode_cards(self.hand.cards)
                    states.append(state_vector)
                
        if GameState.from_game(self.moskaGame).as_vector(normalize=False) != state.as_vector(normalize=False):
            raise Exception("State changed during get_possible_next_states")
        predictions = self.moskaGame.model_predict(np.array(states, dtype=np.float32))
        #predictions = self.model.predict(np.array(states),verbose=0)
        return plays, states, predictions
    
    
    def choose_move(self, playable: List[str]) -> str:
        self.plog.info("Choosing move...")
        self.plog.debug(f"Triumph: {self.moskaGame.triumph_card}")
        self.plog.debug(f"Hand: {self.hand}")
        self.plog.debug(f"Table: {self.moskaGame.cards_to_fall}")
        self.plog.debug(f"Fell: {self.moskaGame.fell_cards}")
        move_scores = {}
        for move in playable:
            move_scores[move] = self.get_prediction(move)
        self.plog.info(f"Move scores: {move_scores}")
        # moveid : (arg, eval)
        self.move_play_scores = move_scores
        if self.requires_graphic:
            print(f"Move scores: {self.move_play_scores.items()}\n")
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
        for play, eval in self.play_fall_from_deck_scores.items():
            if len(play) == 2 and play[0] == deck_card:
                return play
        #return (deck_card, random.choice(self.moskaGame.cards_to_fall))
    
    def end_turn(self) -> List[Card]:
        out = self.move_play_scores["EndTurn"][0]
        return out
    
    
                
        