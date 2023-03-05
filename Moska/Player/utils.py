from typing import Callable, Dict, List, Tuple
import numpy as np
from Moska.Game.Deck import Card
from ..Game import utils

class Assignment:
    """ An assignment is a mapping from cards in the hand to cards on the table.
    Two assignments are equal if the same cards are played to the same cards, regardless of order.
    """
    def __init__(self, inds : Tuple[int]):
        self.inds = inds
        self._hand_inds = self.inds[::2]
        self._table_inds = self.inds[1::2]
    
    def __eq__(self, other):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        return set(self._hand_inds) == set(other._hand_inds) and set(self._table_inds) == set(other._table_inds)
    
    def __repr__(self):
        return f"Assignment({self.inds})"
    
    def __hash__(self):
        """ Two assignments are equal if the same cards are played to the same cards, regardless of order."""
        #return hash(frozenset(self._hand_inds)) + hash(frozenset(self._table_inds))
        return hash(tuple(sorted(list(self._hand_inds)) + sorted(list(self._table_inds))))

def _map_to_list(card : Card, to : List[Card], triumph : str) -> List[Card]:
    """ Return a list of cards, that the input card can fall from `to` list of cards.
    """
    return [c for c in to if utils.check_can_fall_card(card,c,triumph)]

def _map_each_to_list(from_ : List[Card] , to : List[Card], triumph : str) -> Dict[Card,List[Card]]:
    """Map each card in hand, to cards on the table, that can be fallen. Returns a dictionary of from_c : List_t pairs.
    """
    # Make a dictionary of 'card-in-hand' : List[card-on-table] pairs, to know what which cards can be fallen with which cards
    can_fall = {}
    for card in from_:
        can_fall[card] = _map_to_list(card,to,triumph)
    return can_fall


def _make_cost_matrix(from_ : List[Card], to : List[Card], triumph : str, scoring : Callable, max_val : int = 100000) -> np.ndarray:
    """ Make a matrix (len(from_) x len(to)), where each index (i,j) is `scoring(from[i],to[j])` if the card from_[i] can fall to to[j].
    Else the value is `max_val`.

    TODO: Reverse this, so bigger values are better, and smaller values are worse.

    """
    can_fall = _map_each_to_list(from_,to,triumph)
    # Initialize the cost matrix (NOTE: Using inf to denote large values does not work for Scipy)
    C = np.full((len(from_),len(to)),max_val)
    #self.plog.info(f"can_fall: {can_fall}")
    for card, falls in can_fall.items():
        # If there are no cards on the table, that card can fall, continue
        if not falls:
            continue
        card_index = from_.index(card)
        fall_indices = [to.index(c) for c in falls]
        scores = [scoring(card,c) for c in falls]
        C[card_index][fall_indices] = scores
    return C

def _get_single_assignments(matrix : np.ndarray) -> List[List[int]]:
    """ Return all single card assignments from the matrix as a list of lists (row, col).
    """
    nz = np.nonzero(matrix)
    inds = zip(nz[0], nz[1])
    inds = [list(i) for i in inds]
    return inds

def _get_assignments(from_ : List[Card], to : List[Card] = [], triumph : str = "", start=[], found_assignments = None, max_num : int = 1000) -> set[Assignment]:
    """ Return a set of found Assignments, containing all possible assignments of cards from the hand to the cards to fall.
    Symmetrical assignments are considered the same when the same cards are played to the same cards, regardless of order.
    
    The assignments (partial matchings) are searched for recursively, in a depth-first manner:
    - Find all single card assignments (row-col pairs where the intersection == 1)
    - Add the assignment to found_assignments
    - Mark the vertices (row and column of the cost_matrix) as visited (0) (played_card = column, hand_card = row)
    - Repeat
    """
    matrix = None
    if not to:
        if not isinstance(from_, np.ndarray):
            raise TypeError("from_ must be a numpy array if to_ is not given")
        matrix = from_
    # Create a set of found assignments, if none is given (first call)
    if not found_assignments:
        found_assignments = set()
    # If no matrix is given, create the matrix, where 1 means that the card can fall, and 0 means it can't
    if matrix is None:
        if not triumph:
            raise ValueError("triumph must be given if from_ is not a cost matrix")
        matrix = _make_cost_matrix(from_ = from_, to = to, triumph = triumph, max_val=0, scoring=lambda hc,tc : 1)
        
    # Find all single card assignments (row-col pairs where the intersection == 1)
    new_assignments = _get_single_assignments(matrix)
    
    # If there are no more assignments, or the max number of states is reached, return the found assignments
    for row, col in new_assignments:
        if len(found_assignments) >= max_num:
            return found_assignments
        og_len = len(found_assignments)
        # Named tuple with a custom __eq__ and __hash__ method
        assignment = Assignment(tuple(start + [row,col]))
        found_assignments.add(assignment)
        # If the assignment was already found, there is no need to recurse deeper, since there could only be more symmetrical assignments
        if len(found_assignments) == og_len:
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
        _get_assignments(from_ = matrix, start = start +[row,col], found_assignments = found_assignments, max_num = max_num)
        # Restore matrix
        matrix[hand_cards,:] = row_vals
        matrix[:,table_cards] = col_vals
    return found_assignments