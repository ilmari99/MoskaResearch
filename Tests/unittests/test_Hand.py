import unittest
import sys
import os
# Add the main directory to the path
sys.path.append(os.path.abspath(".\\"))
from Moska.Game.Hand import MoskaHand
from Moska.Game.Game import MoskaGame
from Moska.Game.Deck import StandardDeck
from Moska.Player.MoskaBot0 import MoskaBot0
#from Moska.han import Hand,Game,Deck

class TestHand(unittest.TestCase):
    game = None
    deck = None
    hand = None
    def setUp(self) -> None:
        players = [MoskaBot0(name="Bot0"),MoskaBot0(name="Bot1"),MoskaBot0(name="Bot2"),MoskaBot0(name="Bot3")]
        self.game = MoskaGame(players=players)
        self.hand = MoskaHand(self.game)
    
    def test_iterating(self):
        for card,itcard in zip(self.hand.cards,self.hand):
            self.assertEqual(card,itcard)
    
    def test_len6(self):
        self.assertTrue(len(self.hand) == len(self.hand.cards) == 6)
        
    def test_copy_is_equal(self):
        chand = self.hand.copy()
        for card,ccard in zip(self.hand,chand):
            self.assertTrue(card.value == ccard.value and card.suit == ccard.suit)
        self.assertTrue(self.hand.moskaGame is chand.moskaGame)
        
    def test_copy_is_deep(self):
        chand = self.hand.copy()
        chand.pop_cards(max_cards=3)
        self.assertTrue(len(chand) == 3 and len(self.hand) == 6)
        
    def test_adding_to_hand(self):
        self.assertEqual(len(self.hand),6)

        cards = self.game.deck.pop_cards(6)
        self.hand.add(cards)
        self.assertTrue(len(self.hand) == 12)

        popped_cards = self.hand.pop_cards(lambda x : x in cards)
        self.assertTrue(len(self.hand) == 6)

        self.hand.add((c for c in cards))
        self.assertTrue(len(self.hand) == 12)

        self.hand.pop_cards(lambda x : x in cards)
        self.assertTrue(len(self.hand) == 6)
            
if __name__ == "__main__":
    unittest.main()