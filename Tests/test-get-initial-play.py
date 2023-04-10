#!/usr/bin/env python3
from dataclasses import dataclass
import itertools
from collections import Counter
import random
from typing import List, Tuple

CARD_VALUES = tuple(range(1,14)) 
CARD_SUITS = ("C","D","H","S") # Clubs, Diamonds, Hearts, Spades
CARD_SUIT_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣'}    #Conversion table

case1_as_tuple = ([(2,"S"), (2,"H"), (2,"C"), (3,"H"),(3,"H"),(3,"C"),(3,"D"),(10,"S"),(10,"H"),(10,"C"),(13,"S"),(13,"C"),(13,"D"),(7,"H"), (7,"D"), (7,"S")],16)
case2_as_tuple = ([(5,"S"), (3,"S"), (2,"C"), (5,"D"), (5,"H"), (3,"H"),(3,"H"),(3,"C"),(3,"D"),(10,"S"),(10,"H"),(10,"C"),(13,"S"),(13,"C"),(13,"D"),(7,"H"), (7,"D"), (7,"S")],20)
case3_as_tuple = ([(2,"H"), (14,"S"), (5,"S"),(5,"C"),(8,"S"),(8,"C"),(8,"H"),(10,"S"),(10,"C")],10)

@dataclass()
class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        return f"{CARD_SUIT_SYMBOLS[self.suit]}{self.value}"
    
    def __hash__(self):
        return hash((self.value, self.suit))
    
def _make_case(ncards_in_hand = 6, fits = 6):
    """ Get a random hand, and a random number that can be played to the table.
    For timing, a large hand is recommended.
    """
    deck = list(Card(val,suit) for val, suit in itertools.product(CARD_VALUES,CARD_SUITS))
    cards = random.sample(deck, ncards_in_hand)
    return cards, min(fits,ncards_in_hand)

def get_initial_plays_brute(cards : List[Card], n : int) -> List[Tuple[Card]]:
    """ See the logic of the brute force solution here: """
    plays = []
    #plays = itertools.chain.from_iterable((itertools.combinations(cards,i) for i in range(1,n + 1)))
    for i in range(1,fits + 1):
        plays += list(itertools.combinations(cards,i))
    legal_plays = []
    count = 0
    for play in plays:
        count += 1
        c = Counter([c.value for c in play])
        if (len(play) == 1 or all((count >= 2 for count in c.values()))):
            legal_plays.append(play)
    return legal_plays

def get_initial_plays_faster(cards : List[Card], n : int) -> List[Tuple[Card]]:
    plays = []
    # Then we speed up, by creating a generator instead of a list
    plays = itertools.chain.from_iterable((itertools.combinations(cards,i) for i in range(1,n + 1)))
    legal_plays = []
    count = 0
    for play in plays:
        count += 1
        c = Counter([c.value for c in play])
        if (len(play) == 1 or all((count >= 2 for count in c.values()))):
            legal_plays.append(play)
    return legal_plays

def get_initial_plays_even_faster(cards : List[Card], n : int) -> List[Tuple[Card]]:
    # And finally, we use a filter function to filter out the illegal plays, instead of a for loop
    plays = []
    plays = itertools.chain.from_iterable((itertools.combinations(cards,i) for i in range(1,n + 1)))
    filter_func = lambda play: (len(play) == 1 or all((count >= 2 for count in Counter([c.value for c in play]).values())))
    return list(filter(filter_func, plays))

def compare_ans(ans1 : List[Tuple[Card]], ans2 : List[Tuple[Card]]) -> bool:
    # Set is unordered, so we can compare the the lists of tuples
    ans1 = set(ans1)
    ans2 = set(ans2)
    if ans1 == ans2:
        return True
    return False

case1 = ([Card(val,suit) for val, suit in case1_as_tuple[0]], case1_as_tuple[1])
case2 = ([Card(val,suit) for val, suit in case2_as_tuple[0]], case2_as_tuple[1])
case3 = ([Card(val,suit) for val, suit in case3_as_tuple[0]], case3_as_tuple[1])

def your_solution(cards : List[Card], fits : int) -> List[Tuple[Card]]:
    """ Write here"""
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
    plays = itertools.chain.from_iterable(play_iterables)
    legal_plays = []
    count = 0
    for play in plays:
        count += 1
        c = Counter([c.value for c in play])
        if (len(play) == 1 or all((count >= 2 for count in c.values()))):
            legal_plays.append(play)
    return legal_plays

import time

for case_i, case in enumerate([case3, case1, case2, _make_case(20,20)]):
    cards, fits = case
    print(f"Case {case_i + 1}:")
    print(f"Cards: {cards}")
    start = time.time()
    ans1 = get_initial_plays_even_faster(cards, fits)
    print(f"Time taken with gen-filter brute force: {time.time() - start}")
    start = time.time()
    ans2 = your_solution(cards, fits)
    print(f"Time taken with your solution: {time.time() - start}")
    is_correct = compare_ans(ans1, ans2)
    if not is_correct:
        print("Incorrect Answer to case: ", case_i + 1)
    print("\n")

    



