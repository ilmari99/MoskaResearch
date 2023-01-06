from __future__ import annotations
from collections import Counter
import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from ..Game.Deck import Card
from ._ScoreCards import _ScoreCards
if TYPE_CHECKING:
    from ..Game.Game import MoskaGame
from .AbstractPlayer import AbstractPlayer
from .PolicyParameters.HeuristicParameters import HeuristicParameters

from scipy.optimize import linear_sum_assignment

class MoskaBot3(AbstractPlayer):
    cost_matrix_max = 10000
    scoring : _ScoreCards = None
    parameters : HeuristicParameters = None
    def __init__(self, moskaGame: MoskaGame = None, name: str = "", delay=10 ** -6, requires_graphic: bool = False, log_level=logging.INFO, log_file="",parameters = {}):
        if not name:
            name = "B3-"
        super().__init__(moskaGame, name, delay, requires_graphic, log_level, log_file)
        self.scoring = _ScoreCards(self,default_method = "counter")
        if not parameters:
            parameters = {
            'fall_card_already_played_value': 0.025, 
            'fall_card_same_value_already_in_hand': -0.0095, 
            'fall_card_card_is_preventing_kopling': 0.0068, 
            'fall_card_deck_card_not_played_to_unique': 0.0036, 
            'fall_card_threshold_at_start': 0.73, 
            'initial_play_quadratic_scaler': 0.0065
            }
        self.parameters = HeuristicParameters(self,method_values=parameters)
    
    
    def _play_move(self) -> Tuple[bool, str]:
        self.scoring.assign_scores_inplace()
        self.scoring._assign_scores(self.moskaGame.card_monitor.cards_fall_dict.keys())
        return super()._play_move()
        
    def choose_move(self, playable: List[str]) -> str:
        # This must be played
        if "InitialPlay" in playable:
            return "InitialPlay"
        #if self is self.moskaGame.get_target_player() and all((pl.ready for pl in self.moskaGame.get_players_condition(lambda x : x.rank is None and x is not self))):
        #    playable.pop("Skip")
        if "PlayToOther" in playable and not self.play_to_target():
            playable.pop(playable.index("PlayToOther"))
        if "PlayToSelf" in playable and not self.play_to_self():
            playable.pop(playable.index("PlayToSelf"))
        if "PlayFallFromHand" in playable and not self.play_fall_card_from_hand():
            playable.pop(playable.index("PlayFallFromHand"))
        # Now plays only contain plays, that change the status of the game, ie. we actually want to play something
        #if len(playable) > 1 and "Skip" in playable:
        #    playable.pop(playable.index("Skip"))
        #if len(playable) > 1 and "EndTurn" in playable:
        #    playable.pop(playable.index("EndTurn"))
        self.plog.info(f"Want and can plays: {playable}")
        scores = self.parameters.choose_move_scores(playable)
        self.plog.info(f"Scores: {scores}")
        best_play = max(scores.items(),key = lambda x : x[1])
        best_plays = [(pl,score) for pl,score in scores.items() if score == best_play[1]]
        best_play = random.choice(best_plays)
        self.plog.info(f"Playing: {best_play[0]} with score {best_play[1]}")
        #play = random.choice(playable)
        return best_play[0]
    
    def end_turn(self) -> List[Card]:
        """Return which cards you want to pick from the table when finishing your turn.
        Default: pick all cards that cannot be fallen.

        Returns:
            list: List of cards to pick from the table
        """
        pick_cards = self.moskaGame.cards_to_fall
        return pick_cards
    
    def _calc_assignment_score_from_hand(self, hcard : Card, tcard : Card) -> float:
        """ Calculate the score of playing hcard (card in hand) to tcard (card on the table).
        The smaller the score, the better."""
        score = hcard.score - tcard.score
        # Scale the score with some value (currently not calculated [1])
        score = self.parameters.fall_card_scale_hand_play_score(hcard,tcard)*score
        return score
    
    def _calc_assignment_score_from_deck(self,deck_card : Card, tcard : Card):
        score = deck_card.score - tcard.score
        score = self.parameters.fall_card_scale_deck_play_score(deck_card,tcard) * score
        return score
    
    def _calc_assignment_score_to_self(self,card_in_hand : Card, card_to_self : Card):
        score = card_in_hand.score - card_to_self.score
        score = self.parameters.to_self_scale_play_score(card_in_hand,card_to_self)*score
        return score  
    
    
    
    def play_fall_card_from_hand(self) -> Dict[Card, Card]:
        """Return a dictionary of card_in_hand : card_in_table -pairs, denoting which card is used to fall which card on the table.
        This function is called when the player has decided to play from their hand.
        
        Fall the cards with the lowest score

        Returns:
            _type_: _description_
        """
        #self.scoring.assign_scores_inplace()
        # Create the cost matrix
        C = self._make_cost_matrix(scoring=self._calc_assignment_score_from_hand)
        self.plog.info(f"Cost matrix:\n {C}")
        
        # Solve a (possibly) rectangular linear sum assignment problem
        # Return a mapping (indices) from hand to table, that is the optimal assignment, ie. minimize the sum of the associated cost with the assignments.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        hand_indices, fall_indices = linear_sum_assignment(C)
        
        # Loop through the optimal indices
        play_cards = {}
        for hand_ind, table_ind in zip(hand_indices,fall_indices):
            hand_card = self.hand.cards[hand_ind]
            table_card = self.moskaGame.cards_to_fall[table_ind]
            # Discard plays which are not good enough
            if C[hand_ind,table_ind] > self.parameters.fall_card_maximum_play_score_from_hand():
                continue
            # Discard cards that are incorrectly mapped (There will be such cards sometimes)
            if self._check_can_fall_card(hand_card,table_card):
                play_cards[hand_card] = table_card
        return play_cards
    
    
    
    def deck_lift_fall_method(self, deck_card: Card) -> Tuple[Card, Card]:
        """A function to determine which card will fall, if a random card from the deck is lifted.
        Function should take a card -instance as argument, and return a pair (card_from_deck , card_on_table) in the same order,
        determining which card on the table to fall.
        
        This function is called, when the player decides to koplata and the koplattu card can fall a card on the table.
        If the koplattu card can't fall any card on the table, then the card is just placed on the table.
        
        This places the deck_card to the card with the smallest score on the table.
        
        Args:
            deck_card (Card): The lifted card from the deck

        Returns:
            tuple(Card,Card): The input card from deck, the card on the table.
        """
        
        # Get a list of cards that we can fall with the deck_card
        mapping = self._map_to_list(deck_card)
        self.scoring._assign_scores([deck_card])
        sm_score = float("inf")
        best_card = mapping[0]
        for card in mapping:
            score = self._calc_assignment_score_from_deck(deck_card,card)
            if score < sm_score:
                sm_score = score
                best_card = card
        return (deck_card,best_card)
    
    def play_to_self(self) -> List[Card]:
        """Which cards from hand to play to table.
        Default play all playable values, except triumphs

        Returns:
            List[Card]: list of cards played to self
        """
        pv = self._playable_values_from_hand()
        
        # Get a mapping from each card in hand, to all cards in hand that can be fallen with the card and can be played to the table
        c_playable = self.hand.copy().pop_cards(lambda x : x.value in pv)
        can_fall = self._map_each_to_list(from_ = self.hand.cards, to=c_playable)
        cards = []
        for card, falls in can_fall.items():
            for fall_card in falls:
                score = self._calc_assignment_score_to_self(card,fall_card)
                if score > self.parameters.to_self_maximum_play_score():
                    continue
                cards.append(fall_card)
        return list(set(cards))
    
    def play_initial(self) -> List[Card]:
        """Return a list of cards to play from hand to an empty table.
        
        Calculates a dictionary with unique values in hand as keys, and the cards in hand with that value as values. The cards are sorted in ascending
        order. For example d[3] : [S3,A3], where S3.score = 4, S3.score = 6
        
        
        Returns:
            List[Card]: _description_
        """
        same_values = {}
        for val in set([c.value for c in self.hand.cards]):
            # A dictionary of value : List[Card], where the cards are sorted in ascending order according to score
            # For example same_values[3] : [S3,A3], where S3.score = 4, S3.score = 6
            same_values[val] = list(sorted(filter(lambda x : x.value == val, self.hand.cards),key=lambda x : x.score))
        fits = self._fits_to_table()
        play_cards = []
        new_play_cards = []
        # Search for play cards, atleast once, and until fits < 2
        while (not play_cards and fits >= 1) or (fits >= 2):
            scores = self._calc_initial_scores_dict(same_values,fits)
            play_val = 0    # Value to play
            play_ncards = 1 # Number of cards to play
            play_score = float("inf")   # The smallest score of the play, when playing ncards with certain value.
            # Loop through scores, which contains 'value : (ncards,score)' pairs, and find the value and number of cards to play.
            for val, res in scores.items():
                ncards, score = res
                # If score is smaller than the currently found smallest score, update the variables.
                if score < play_score:
                    play_val = val
                    play_ncards = ncards
                    play_score = score
            # If there are cards chosen to be played AND we have only selected 1 new card to play, OR we only chose to play 1 card on the first loop.
            # the move is illegal, and we must stop the search,
            # because we can only play a single card, or multiple pairs or greater.
            if (play_cards and play_ncards == 1) or (len(play_cards) == 1):
                break
            # Once we know the new cards can be played, we remove the cards from the dictionary, and decrement fits variable
            choose_from = same_values[play_val]
            # Take ncards from the cards
            new_play_cards = choose_from[:None if play_ncards == len(choose_from) else play_ncards]
            play_cards += new_play_cards
            # remove from same_values dict (choose from points to it)
            [choose_from.remove(card) for card in new_play_cards]
            fits -= play_ncards
        self.plog.info(f"INITIAL: Chose: {play_cards}")
        return play_cards
    
    def _calc_initial_scores_dict(self,same_values : dict[int,List[Card]], fits : int) -> Dict[int,Tuple[int,float]]:
        """Return a dictionary of value : (ncards, score) -pairs from every unique value in hand.
        

        Args:
            same_values (dict[int,List[Card]]): _description_
            fits (int): _description_

        Returns:
            Dict[int,Tuple[int,float]]: _description_
        """
        scores = {}
        # Loop through the same_values dictionary: Dict[int,List[Card]]
        for val,cards in same_values.items():
            if not cards:
                continue
            cards = list(cards)
            # Calculate a list of scores, that are achieved by playing i (1...min(fits,len(cards))) cards.
            # The cards in cards are supposed to be in ascending order
            ncard_scores = [self._calc_initial_play_score(cards,ncards) for ncards in range(1,min(fits,len(cards))+1)]
            # Store the number of cards, and the score corresponding with that amount of played cards
            sm_score = min(ncard_scores)
            ncards = ncard_scores.index(sm_score) + 1
            scores[val] = (ncards,sm_score)
        return scores
    
    def _calc_initial_play_score(self,cards : List[Card],fits : int) -> float:
        """Calculate the score for playing 'fits' first cards from 'cards'.
        
        Args:
            cards (List[Card]): _description_
            fits (int): _description_

        Returns:
            float: _description_
        """
        ncards = min(fits,len(cards))
        # Get ncards first cards from 'cards'
        play_cards = cards[0:None if ncards == len(cards) else ncards]
        cards_score = sum([c.score for c in play_cards]) / ncards
        cards_score = self.parameters.initial_play_scale_score(play_cards) * cards_score
        # Return the adjusted average score
        return cards_score
    
    def play_to_target(self) -> List[Card]:
        """ Return a list of cards, that will be played to target.
        This function is called, when there are cards on the table, and you can play cards to a target
        
        This method is meant to be overwriteable.
        Default: Play all playable values that fit.
        """
        playable_values = self._playable_values_from_hand()
        play_cards = []
        if playable_values:
            chand = self.hand.copy()
            play_cards = chand.pop_cards(cond=lambda x : x.value in playable_values and (x.score < 10 or len(self.moskaGame.deck) <= 0), max_cards = self._fits_to_table())
        return play_cards