#!/usr/bin/env python3
from dataclasses import dataclass
import itertools
from collections import Counter
import random
from typing import List, Tuple

CARD_VALUES = tuple(range(1,14)) 
CARD_SUITS = ("C","D","H","S") # Clubs, Diamonds, Hearts, Spades
CARD_SUIT_SYMBOLS = {"S":'♠', "D":'♦',"H": '♥',"C": '♣'}    #Conversion table


@dataclass
class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        return f"{CARD_SUIT_SYMBOLS[self.suit]}{self.value}"


def get_initial_plays_faster(cards : List[Card], n : int) -> List[Tuple[Card]]:
    plays = []
    plays = itertools.chain.from_iterable((itertools.combinations(cards,i) for i in range(1,n + 1)))
    #for i in range(1,fits + 1):
    #    plays += list(itertools.combinations(cards,i))
    legal_plays = []
    count = 0
    for play in plays:
        count += 1
        c = Counter([c.value for c in play])
        if (len(play) == 1 or all((count >= 2 for count in c.values()))):
            legal_plays.append(play)
    print(f"Compared {count} plays, found {len(legal_plays)} legal plays")
    return legal_plays

def get_initial_plays_even_faster(cards : List[Card], n : int) -> List[Tuple[Card]]:
    plays = []
    plays = itertools.chain.from_iterable((itertools.combinations(cards,i) for i in range(1,n + 1)))
    filter_func = lambda play: (len(play) == 1 or all((count >= 2 for count in Counter([c.value for c in play]).values())))
    return list(filter(filter_func, plays))

def get_initial_plays_brute(cards : List[Card], n : int) -> List[Tuple[Card]]:
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
    print(f"Compared {count} plays, found {len(legal_plays)} legal plays")
    return legal_plays

def make_case(ncards_in_hand = 6, fits = 6, get_pre_made_case = False):
    PRE_MADE_CASES = [
        ([(2,"S"), (2,"H"), (3,"H"),(3,"H"),(3,"C"),(3,"D"),(10,"S"),(10,"H"),(10,"C")],9),
        ([(2,"S"), (2,"H"),(3,"H"),(3,"C"),(3,"D"),(10,"S"),(10,"H"),(10,"C")],7),
        ([(2,"S"), (2,"H"), (2,"C"), (3,"H"),(3,"H"),(3,"C"),(3,"D"),(10,"S"),(10,"H"),(10,"C"),(13,"S"),(13,"C"),(13,"D"),(7,"H"), (7,"D"), (7,"S")],16),
    ]
    if isinstance(get_pre_made_case, int):
        cards = [Card(val,suit) for val, suit in PRE_MADE_CASES[get_pre_made_case][0]]
        fits = PRE_MADE_CASES[get_pre_made_case][1]
    else:
        deck = list(Card(val,suit) for val, suit in itertools.product(CARD_VALUES,CARD_SUITS))
        cards = random.sample(deck, ncards_in_hand)
    return cards, min(fits,ncards_in_hand)

import timeit
cards, fits = make_case(6,8, get_pre_made_case=2)
print(cards)
print(timeit.timeit("get_initial_plays_brute(cards, fits)", number=10, globals=globals()))
print(timeit.timeit("get_initial_plays_faster(cards, fits)", number=10, globals=globals()))
print(timeit.timeit("get_initial_plays_even_faster(cards, fits)", number=10, globals=globals()))

    



