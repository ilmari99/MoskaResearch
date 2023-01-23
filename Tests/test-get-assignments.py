import functools
import itertools
import random
import time
from typing import Callable, Dict, List, Set, Tuple
import numpy as np

CARD_VALUES = tuple(range(1,14)) 
CARD_SUITS = ("C","D","H","S") # Clubs, Diamonds, Hearts, Spades
CARD_SUIT_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣'}    #Conversion table


class Assignment:
    def __init__(self, inds : Tuple[int],hash_method = "sortedtuple"):
        self.inds = inds
        self._hand_inds = self.inds[::2]
        self._table_inds = self.inds[1::2]
        self.hash_method = hash_method
    
    def __eq__(self, other):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        return tuple(sorted(list(self._hand_inds)) + sorted(list(self._table_inds))) == tuple(sorted(list(other._hand_inds)) + sorted(list(other._table_inds)))
        #return set(self._hand_inds) == set(other._hand_inds) and set(self._table_inds) == set(other._table_inds)
    
    def __repr__(self):
        return f"{self.inds}"
    
    def __hash__(self):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        if self.hash_method == "frozenset":
            return hash(frozenset(self._hand_inds)) + hash(frozenset(self._table_inds))
        elif self.hash_method == "sortedtuple":
            return hash(tuple(sorted(list(self._hand_inds)) + sorted(list(self._table_inds))))
        #return hash(tuple(sorted(list(self._hand_inds)) + sorted(list(self._table_inds))))

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        return f"{CARD_SUIT_SYMBOLS[self.suit]}{self.value}"
    
    def __hash__(self):
        return hash((self.value, self.suit))


def check_can_fall_card(played_card : Card, fall_card : Card,triumph : str) -> bool:
    """Returns true, if the played_card, can fall the fall_card.
    The played card can fall fall_card, if:
    - The played card has the same suit and is greater than fall_card
    - If the played_card is triumph suit, and the fall_card is not.

    Args:
        played_card (Card): The card played from hand
        fall_card (Card): The card on the table
        triumph (str): The triumph suit of the current game

    Returns:
        bool: True if played_card can fall fall_card, false otherwise
    """
    success = False
    # Jos kortit ovat samaa maata ja pelattu kortti on suurempi
    if played_card.suit == fall_card.suit and played_card.value > fall_card.value:
        success = True
    # Jos pelattu kortti on valttia, ja kaadettava kortti ei ole valttia
    elif played_card.suit == triumph and fall_card.suit != triumph:
            success = True
    return success

def _map_each_to_list(from_, to):
    """Map each card in hand, to cards on the table, that can be fallen.
    Returns a dictionary

    Returns:
        _type_: _description_
    """
    # Make a dictionary of 'card-in-hand' : List[card-on-table] pairs, to know what which cards can be fallen with which cards
    can_fall = {}
    for card in from_:
        can_fall[card] = [c for c in to if check_can_fall_card(card,c,"H")]
    return can_fall

def _make_matrix(from_, to):
    #self.plog.debug(f"Making cost matrix from {from_} to {to}")
    scoring = lambda hc,tc : 1
    # Some functions cant handle infs, so use a large value instead
    max_val = 0
    can_fall = _map_each_to_list(from_,to)
    # Initialize the cost matrix (NOTE: Using inf to denote large values does not work for Scipy)
    cm = np.full((len(from_),len(to)),max_val)
    #self.plog.info(f"can_fall: {can_fall}")
    for card, falls in can_fall.items():
        # If there are no cards on the table, that card can fall, continue
        if not falls:
            continue
        card_index = from_.index(card)
        fall_indices = [to.index(c) for c in falls]
        scores = [scoring(card,c) for c in falls]
        cm[card_index][fall_indices] = scores
    return cm

def _get_single_assignments(matrix) -> List[List[int]]:
    """ Return all single card assignments from the matrix as a list of lists (row, col).
    """
    nz = np.nonzero(matrix)
    inds = zip(nz[0], nz[1])
    inds = [list(i) for i in inds]
    return inds

def _get_assignments(matrix : np.ndarray, start = [], found_assignments = None, max_num_states = 1000) -> Set[Assignment]:
        """ Return a set of found Assignments, containing all possible assignments of cards from the hand to the cards to fall.
        Symmetrical assignments are considered the same: where the same cards are played to the same cards, regardless of order.
        
        The assignments (partial matchings) are searched for recursively, in a depth-first manner:
        - Find all single card assignments (row-col pairs where the intersection == 1)
        - Add the assignment to found_assignments
        - Mark the vertices (row and column of the cost_matrix) as visited (0) (played_card = column, hand_card = row)
        - Repeat
        """
        # Create a set of found assignments, if none is given (first call)
        if not found_assignments:
            found_assignments = set()
            
        # Find all single card assignments (row-col pairs where the intersection == 1)
        new_assignments = _get_single_assignments(matrix)
        
        for row, col in new_assignments:
            if len(found_assignments) >= max_num_states:
                return found_assignments
            # Named tuple with a custom __eq__ and __hash__ method
            assignment = Assignment(tuple(start + [row,col]),hash_method="sortedtuple")
            found_assignments.add(assignment)
            # Get the visited cards
            # The cards in hand are even (0,2...), the cards on the table are odd (1,3...)
            #hand_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 0]
            #table_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 1]
            hand_cards = assignment._hand_inds
            table_cards = assignment._table_inds
            # Store the values, to restore them later
            row_vals = matrix[hand_cards,:]
            col_vals = matrix[:,table_cards]
            # Mark the cards as visited (0)
            matrix[hand_cards,:] = 0
            matrix[:,table_cards] = 0
            # Find the next assignments, which adds the new assignments to the found_assignments
            _get_assignments(matrix, start = start +[row,col], found_assignments = found_assignments, max_num_states = max_num_states)
            # Restore matrix
            matrix[hand_cards,:] = row_vals
            matrix[:,table_cards] = col_vals
        if not start:
            print(f"Found {len(found_assignments)} assignments")
        return found_assignments
    
def _get_assignments_custom(matrix : np.ndarray, start = [], found_assignments = None, max_num_states = 1000000000000000) -> Set[Assignment]:
        """ Return a set of found Assignments, containing all possible assignments of cards from the hand to the cards to fall.
        Symmetrical assignments are considered the same: where the same cards are played to the same cards, regardless of order.
        
        The assignments (partial matchings) are searched for recursively, in a depth-first manner:
        - Find all single card assignments (row-col pairs where the intersection == 1)
        - Add the assignment to found_assignments
        - Mark the vertices (row and column of the cost_matrix) as visited (0) (played_card = column, hand_card = row)
        - Repeat
        """
        # Create a set of found assignments, if none is given (first call)
        if not found_assignments:
            found_assignments = set()
            
        # Find all single card assignments (row-col pairs where the intersection == 1)
        new_assignments = _get_single_assignments(matrix)
        
        if not new_assignments:
            return set()
        
        for row, col in new_assignments:
            if len(found_assignments) >= max_num_states:
                return found_assignments
            # Named tuple with a custom __eq__ and __hash__ method
            assignment = Assignment(tuple(start + [row,col]), hash_method="sortedtuple")
            orig_len = len(found_assignments)
            found_assignments.add(assignment)
            if len(found_assignments) == orig_len:
                continue
            # Get the visited cards
            # The cards in hand are even (0,2...), the cards on the table are odd (1,3...)
            #hand_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 0]
            #table_cards = [c for i,c in enumerate(start + [row,col]) if i % 2 == 1]
            hand_cards = assignment._hand_inds
            table_cards = assignment._table_inds
            # Store the values, to restore them later
            row_vals = matrix[hand_cards,:]
            col_vals = matrix[:,table_cards]
            # Mark the cards as visited (0)
            matrix[hand_cards,:] = 0
            matrix[:,table_cards] = 0
            # Find the next assignments, which adds the new assignments to the found_assignments
            _get_assignments_custom(matrix, start = start +[row,col], found_assignments = found_assignments, max_num_states = max_num_states)
            # Restore matrix
            matrix[hand_cards,:] = row_vals
            matrix[:,table_cards] = col_vals
        if not start:
            print(f"Found {len(found_assignments)} assignments")
        return found_assignments

def create_mat(n):
    """ Create a matrix of size n x n, with 1s randomly """
    mat = np.zeros((n,n))
    for i in range(n):
        mat[random.randint(0,n-1),random.randint(0,n-1)] = 1
    print(mat.shape)
    print()
    return mat

def test_complexity():
    best,others = big_o.big_o(_get_assignments_custom, create_mat, n_repeats=1,min_n=2, max_n=24, n_measures = 22)
    print(best)
    for class_, residuals in others.items():
        print('{!s:<60s}    (res: {:.2G})'.format(class_, residuals))

import cProfile
import big_o
if __name__ == "__main__":
    for i in range(1):
        deck = [Card(val,s) for val,s in itertools.product(CARD_VALUES, CARD_SUITS)]
        tot_cards = random.sample(deck, 30)
        hand = tot_cards[0:len(tot_cards)//2]
        table = tot_cards[len(tot_cards)//2:]
        print(f"Triumph: {CARD_SUIT_SYMBOLS['H']}")
        print(f"Hand: {hand}")
        print(f"Table: {table}")
        #cProfile.run('_get_assignments(_make_matrix(hand,table),max_num_states = 1000000)', sort='tottime')
        start_mat = time.time()
        mat = _make_matrix(hand,table)
        print(f"Time to make matrix: {time.time() - start_mat} seconds")
        start_assignments = time.time()
        #assignments = _get_assignments(mat,max_num_states = 1000000)
        print(f"Time to find assignments with old method: {time.time() - start_assignments} seconds")
        start_custom_assignments = time.time()
        custom_assignments = _get_assignments_custom(mat,max_num_states = 1000000)
        print(f"Time to find assignments with new method: {time.time() - start_custom_assignments} seconds")
        #if assignments == custom_assignments:
        #    print("Assignments ARE equal")
        #else:
        #    print("Assignments ARE NOT equal")
        #    print(assignments.difference(custom_assignments))
        #print(f"Assignments: ")
        #for ass in assignments:
        #    print(ass)
        print(f"Number of possible assignments: {len(custom_assignments)}")