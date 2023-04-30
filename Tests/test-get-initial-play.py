#!/usr/bin/env python3
from dataclasses import dataclass
from functools import lru_cache
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
    
class CardPlay:
    """ A class to represent a play of cards to the table. 
    """
    def __init__(self, cards : List[Card]):
        self.cards = cards

    def __hash__(self):
        return hash(tuple(self.cards))
    
    def __eq__(self, other):
        return set(self.cards) == set(other.cards)
    
    def __repr__(self):
        return f"{self.cards}"
    
REFERENCE_DECK = list(Card(val,suit) for val, suit in itertools.product(CARD_VALUES,CARD_SUITS))
    
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
    def check_legal(play):
        return (len(play) == 1 or all((count >= 2 for count in Counter([c.value for c in play]).values())))
    legal_plays = list(filter(check_legal, plays))
    return list(set(legal_plays))

def int_partition(n):
    """
    Generate all partitions of integer n (>= 0) as tuples in weakly descending order.
    """
    a = [0 for i in range(n + 1)]
    k = 1
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while x <= y:
            a[k] = x
            y -= x
            k += 1
        a[k] = x + y
        yield a[:k + 1]

def new_solution(cards : List[Card], fits : int) -> List[Tuple[Card]]:
    single_solutions = itertools.combinations(cards,1)
    og_counter = Counter([c.value for c in cards])
    cards = [c for c in cards if og_counter[c.value] >= 2]
    #card_sets = []
    #for val, count in og_counter.items():
    #    if count >= 2:
    #        card_sets.append([c for c in cards if c.value == val])
    card_sets = [[c for c in cards if c.value == val] for val, count in og_counter.items() if count >= 2]

    legal_plays = list(single_solutions)
    # Take all combinations of atleast two cards from each set
    # Store in a dictionary, where the first key is the cards value, and the second key is the length of the play
    cards_set_combinations = {}
    for i in range(1,len(card_sets) + 1):
        value = card_sets[i - 1][0].value
        cards_set_combinations[value] = {}
        for len_play in range(2, min(fits, len(card_sets[i - 1]))+1):
            plays = list(itertools.combinations(card_sets[i - 1],len_play))
            if len(plays) > 0:
                cards_set_combinations[value][len_play] = plays

    # Now we have a dictionary of each card value, and all the pure plays of that value
    # cards_set_combinations = {value(int) : {len_play(int) : plays(List)}}

    # Now we want to find all the combinations of those plays
    # We do this, by sort of tree searching, where we start with a value, and add all of its pure plays to the list of plays
    # Then for each of those plays, we go to all other values, and combine all of their pure plays with the current play and them to the list of plays
    # Then for each of those plays, we go to all other values, and combine all of their pure plays with the current play and them to the list of plays
    # And so on, until we have gone through all values
    def get_play_combinations(play,visited = set(), started_with = set()):
        """ Return a list of combinations from the cards_set_combinations dictionary. The input is a tuple play,
        and this function returns all plays, that can be combined with the input play, that do not share values with the input play.
        """
        play = list(play)
        if len(play) >= fits:
            return [play]
        if not visited:
            visited = set((c.value for c in play))
        combined_plays = [play]
        for val, plays in cards_set_combinations.items():
            if val not in visited:
                for len_play, plays in plays.items():
                    if len_play + len(play) > fits:
                        continue
                    for p in plays:
                        if p in started_with:
                            continue
                        visited.add(val)
                        old_visited = visited.copy()
                        combs = get_play_combinations(tuple(list(play) + list(p)),visited,started_with)
                        visited = old_visited
                        combined_plays += combs
        return combined_plays
    
    started_with = set()
    # Now we have a function that can return all combinations of plays, that do not share values, from some starting play
    for val, plays in cards_set_combinations.items():
        for len_play, plays in plays.items():
            for play in plays:
                started_with.add(tuple(play))
                combs = get_play_combinations(tuple(play),started_with=started_with)
                legal_plays += [tuple(c) for c in combs]
    legal_plays = list(set(legal_plays))
    return legal_plays



def are_plays_equal(play1 : CardPlay, play2 : CardPlay) -> bool:
    """ Return True if the two plays are equal, and False otherwise. """
    if len(play1.cards) != len(play2.cards):
        return False
    if play1 != play2:
        return False
    return True

def are_lists_equal(list1 : List[CardPlay], list2 : List[CardPlay]) -> bool:
    """ Return True if the two lists are equal, and False otherwise. """
    if len(list1) != len(list2):
        return False
    for play1 in list1:
        if not any((are_plays_equal(play1, play2) for play2 in list2)):
            return False
    return True
    

def compare_ans(ans1 : List[Tuple[Card]], ans2 : List[Tuple[Card]]) -> bool:
    """ Analyze the two answers, and print the differences."""
    if not isinstance(ans1[0], CardPlay):
        ans1 = [CardPlay(play) for play in ans1]
        ans2 = [CardPlay(play) for play in ans2]
    if are_lists_equal(ans1, ans2):
        print("The two answers are equal")
        return True
    else:
        print("The two answers are not equal")
        only_in_ans1 = [play for play in ans1 if play not in ans2]
        only_in_ans2 = [play for play in ans2 if play not in ans1]
        print(f"Only in ans1: {only_in_ans1}")
        print(f"Only in ans2: {only_in_ans2}")
        ans1_duplicates = [play for play in ans1 if ans1.count(play) > 1]
        ans2_duplicates = [play for play in ans2 if ans2.count(play) > 1]
        print(f"Ans1 duplicates: {ans1_duplicates}")
        print(f"Ans2 duplicates: {ans2_duplicates}")
        return False

case1 = ([Card(val,suit) for val, suit in case1_as_tuple[0]], case1_as_tuple[1])
case2 = ([Card(val,suit) for val, suit in case2_as_tuple[0]], case2_as_tuple[1])
case3 = ([Card(val,suit) for val, suit in case3_as_tuple[0]], case3_as_tuple[1])
case4 = ([Card(val,suit) for val, suit in [(12,"D"), (2,"H"), (2,"D"), (3,"H"), (12,"H"), (12,"S"),(3,"D")]], 6)
case5 = ([Card(val,suit) for val, suit in [(2,"H"), (3,"H"),(2,"D"),(3,"D"),(2,"S")]], 4)

play1_list = []
play2_list = []
# Test the CardPlay class
play1 = CardPlay([Card(2,"H"), Card(2,"D"), Card(2,"S")])
play2 = CardPlay([Card(2,"D"), Card(2,"H"), Card(2,"S")])
print(f"Test1 two equal plays in different order (True): {are_plays_equal(play1, play2)}")
play1_list.append(play1)
play2_list.append(play2)
play1 = CardPlay([Card(2,"H"), Card(2,"D"), Card(3,"S")])
play2 = CardPlay([Card(2,"D"), Card(2,"H"), Card(3,"S")])
print(f"Test2 two equal plays in different order (True): {are_plays_equal(play1, play2)}")
play1_list.append(play1)
play2_list.append(play2)

# Test the are_lists_equal function. Here they should have the same elements, but in different order
print(f"Test3 two lists of equal plays in different order (True): {are_lists_equal(play1_list, play2_list)}")
print(compare_ans(play1_list, play2_list))

# Add a duplicate to the second list
play2_list.append(play2)
print(f"Test4 two lists of equal plays but with a duplicate (False): {are_lists_equal(play1_list, play2_list)}")
print(compare_ans(play1_list, play2_list))


import time
#case3, case1, case2, _make_case(20,20), 
for case_i, case in enumerate([case4, case5, case1,case2,case3]):#, case1, case2]):
    cards, fits = case
    print(f"Case {case_i + 1}:")
    print(f"Cards: {cards}")
    start = time.time()
    ans1 = your_solution(cards, fits)
    print(f"Time taken with gen-filter brute force {len(ans1)}: {time.time() - start}")
    start = time.time()
    ans2 = new_solution(cards, fits)
    print(f"Time taken with your solution {len(ans2)}: {time.time() - start}")
    is_eq = compare_ans(ans1, ans2)
    print(f"The answers are equal: {is_eq}\n")
    #if True or not is_eq:
    #    print(f"Correct answer: {ans1}")
    #    print(f"Your answer: {ans2}")
    #    compare_ans(ans1, ans2)

    



