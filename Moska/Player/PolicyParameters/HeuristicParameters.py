import random
from ..AbstractPlayer import AbstractPlayer
from ...Game.Deck import Card

from typing import TYPE_CHECKING, Any, Dict,Callable, List



class HeuristicParameters():
    method_values = {}
    player : AbstractPlayer = None
    def __init__(self, player: AbstractPlayer,method_values : Dict[str,float] = {}) -> None:
        #"""
        self.player = player
        def_method_values = {'fall_card_already_played_value': -0.38, 
         'fall_card_same_value_already_in_hand': 0.072, 
         'fall_card_card_is_preventing_kopling': -0.29, 
         'fall_card_deck_card_not_played_to_unique': 0.336, 
         'fall_card_threshold_at_start': 30.44, 
         'initial_play_quadratic_scaler': 0.61
         }
         #"""
         #self.method_values = {}
         # If there are no method values, use the default values
        if not method_values:
            method_values = def_method_values
        # If method_values is a string, and it is "random", then use random values where the random values are (0,2) times the default value
        elif isinstance(method_values,str) and method_values == "random":
            method_values = {}
            for key,value in def_method_values.items():
                method_values[key] = value + random.uniform(-value,value)
        # If method_values is a dictionary, then use the default values, and update them with the given values
        elif isinstance(method_values,dict):
            method_values = {**def_method_values,**method_values}
        self.method_values = method_values
        return
    
    def _adjust_for_missing_cards(self, cards: List[Card], most_falls,lifted = 0) -> float:
        """ Adjust score for missing cards (each card missing from hand is as valuable as the most falling card)
        """
        missing_from_hand = 6 - len(cards) - lifted
        return max(most_falls * missing_from_hand,0)
        
    def _lift_n_from_deck(self, cards : List[Card]) -> float:
        """ Return how many cards must be lifted from the deck, given current hand"""
        missing = 6 - len(cards)
        liftn = 0
        if missing > 0 and len(self.player.moskaGame.deck) > 0:
            liftn = min(missing, len(self.player.moskaGame.deck))
        return liftn
    
    def _e_score_from_lifted(self, e_lifted : float, n : int) -> float:
        """ Return the total excpected score of lifted cards """
        return e_lifted * n
    
    def _calculate_score(self, cards_after_play : List[Card], lifted_from_deck : int, most_falls : int, e_lifted : float) -> float:
        """ Evaluate the hand after playing, or the excpected value of the hand"""
        try:
            sc = sum((c.score for c in cards_after_play)) + self._adjust_for_missing_cards(cards_after_play,most_falls,lifted=lifted_from_deck) + self._e_score_from_lifted(e_lifted, lifted_from_deck)
        except TypeError as te:
            print([c.score for c in cards_after_play])
            raise TypeError(te)
        try:
            sc = sc / (len(cards_after_play) + lifted_from_deck)
        except ZeroDivisionError as zde:
            sc = float("inf")
        return sc
    
    def choose_move_scores(self,moves : List[str]) -> float:
        """ Evaluate the expected score after playing a move.
        This calls the corresponding method from the player, to see which cards the player is going to play"""
        
        e_lifted = self.expected_value_from_lift()
        most_falls = max((len(falls) for card, falls in self.player.moskaGame.card_monitor.cards_fall_dict.items()))
        self.player.plog.info(f"Expected score from deck: {e_lifted}")
        self.player.plog.info(f"Most falling card: {most_falls}")
        move_scores = {}
        for move in moves:
            cards_after_play = self.player.hand.cards
            liftn = 0
            
            # In this if- structure, we define 'cards_after_play' and 'liftn'
            
            if move == "Skip":
                # The score of hand stays the same
                cards_after_play = self.player.hand.cards
            
            elif move == "EndTurn":
                # The cards returned by player.end_turn() are lifted, and if necessary, filled from deck
                cards_after_play = self.player.hand.cards + self.player.end_turn()
                liftn = self._lift_n_from_deck(cards_after_play)
            
            elif move == "PlayToOther":
                played_cards = self.player.play_to_target()
                cards_after_play = list(set(self.player.hand.cards).difference(played_cards))
                liftn = self._lift_n_from_deck(cards_after_play)
                
            elif move == "PlayFallFromHand":
                # TODO: Bad
                played_cards = list(self.player.play_fall_card_from_hand().keys())
                cards_after_play = list(set(self.player.hand.cards).difference(played_cards))
                liftn = self._lift_n_from_deck(cards_after_play)
            
            elif move == "PlayToSelf":
                played_cards = self.player.play_to_self()
                cards_after_play = list(set(self.player.hand.cards).difference(played_cards))
                liftn = self._lift_n_from_deck(cards_after_play)
            else:
                liftn = self._lift_n_from_deck(cards_after_play)
                
            score = self._calculate_score(cards_after_play, liftn,most_falls,e_lifted)
            move_scores[move] = score
        return move_scores

    def expected_value_from_lift(self):
        """ Calculate the expected score of a card that is lifted from the deck.
        Check which cards location we know (Cards in hand + other players known cards).
        >> The remaining cards are either in deck, or in players hands.
        Then calculate the total fall score and divide by the number of cards whose location is not known,
        
        """
        game = self.player.moskaGame
        cards_not_in_deck = self.player.hand.copy().cards + game.fell_cards
        for pl, cards in game.card_monitor.player_cards.items():
            if pl == self.player.name:
                continue
            for card in cards:
                if card != Card(-1,"X"):
                    cards_not_in_deck.append(card)
        self.player.plog.debug(f"Cards NOT in deck: {len(cards_not_in_deck)}")
        cards_possibly_in_deck = set(game.card_monitor.cards_fall_dict.keys()).difference(cards_not_in_deck)
        self.player.plog.debug(f"Cards possibly in deck: {len(cards_possibly_in_deck)}")
        try:
            total_possible_falls = sum((c.score for c in cards_possibly_in_deck))
        except TypeError as te:
            print(f"Cards possibly in deck: {cards_possibly_in_deck}")
            print([c.score for c in cards_possibly_in_deck])
            raise TypeError(te)
        try:
            e_lifted = total_possible_falls / len(cards_possibly_in_deck)
        except ZeroDivisionError as ze:
            e_lifted = 0
        return e_lifted
    
    def fall_card_scale_hand_play_score(self, hcard: Card, tcard: Card, **kwargs) -> float:
        scale = 1
        # If value is already played
        if hcard.value in set([c.value for c in self.player.moskaGame.cards_to_fall]):
            scale += self.method_values["fall_card_already_played_value"]
        # If I already have same values in hand, it is perhaps easier to get rid of the card if lifted -> Increase the score
        if tcard.value in set([c.value for c in self.player.hand.cards]):
            scale += self.method_values["fall_card_same_value_already_in_hand"]
        # If the card has been kopled and is preventing us from kopling again
        if tcard.kopled and len(self.player.moskaGame.deck) > 0:
            scale += self.method_values["fall_card_card_is_preventing_kopling"]
        scale = scale*(hcard.score + tcard.score)/(hcard.score - tcard.score)
        #scale = scale*(hcard.score - tcard.score)/tcard.score
        return scale
    
    def fall_card_scale_deck_play_score(self, deck_card: Card, tcard: Card) -> float:
        scale = 1
        # Check if tcard can be fallen with another card
        mapping = self.player._map_each_to_list()
        can_fall_with_other_cards = False
        for hcard, fall_cards in mapping.items():
            if tcard in fall_cards:
                can_fall_with_other_cards = True
                break
        if can_fall_with_other_cards:
            scale += self.method_values["fall_card_deck_card_not_played_to_unique"]
        scale = scale*(deck_card.score + tcard.score) / (deck_card.score - tcard.score)
        #scale = scale*(hcard.score - tcard.score)/(tcard.score + deck_card.score)
        return scale
    
    def to_self_scale_play_score(self, card_in_hand: Card, card_to_self: Card):
        scale = 1#(card_in_hand.score - card_to_self.score)/card_to_self.score
        scale = scale*(card_in_hand.score + card_to_self.score) / (card_in_hand.score - card_to_self.score)
        return scale
    
    def fall_card_maximum_play_score_from_hand(self, **kwargs):
        threshold = ((self.method_values["fall_card_threshold_at_start"]-52)/52)*len(self.player.moskaGame.deck) + 52
        return threshold
    
    def to_self_maximum_play_score(self):
        threshold = -len(self.player.moskaGame.deck) + 52
        return threshold
    
    def initial_play_scale_score(self, play_cards: List[Card]):
        return self.method_values["initial_play_quadratic_scaler"] * len(play_cards) + 1 - self.method_values["initial_play_quadratic_scaler"]