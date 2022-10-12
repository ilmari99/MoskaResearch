import unittest
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/")
SCRIPT_DIR = "\\".join(SCRIPT_DIR[0:-2])
#print(SCRIPT_DIR)
sys.path.insert(1,os.path.dirname(SCRIPT_DIR))
from Moska import Hand,Game,Deck

class TestHand(unittest.TestCase):
    game = None
    deck = None
    hand = None
    def setUp(self) -> None:
        self.deck = Deck.StandardDeck()
        self.game = Game.MoskaGame(deck=self.deck)
        self.hand = Hand.MoskaHand(self.game)
    
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
        cards = self.deck.pop_cards(6)
        self.hand.add(cards)
        self.assertTrue(len(self.hand) == 12)
        self.hand.pop_cards(lambda x : x in cards)
        self.assertTrue(len(self.hand) == 6)
        self.hand.add((c for c in cards))
        self.assertTrue(len(self.hand) == 12)
        self.hand.pop_cards(lambda x : x in cards)
        self.assertTrue(len(self.hand) == 6)
            
if __name__ == "__main__":
    unittest.main()