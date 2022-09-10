import unittest
import copy
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/")
SCRIPT_DIR = "\\".join(SCRIPT_DIR[0:-2])
#print(SCRIPT_DIR)
sys.path.insert(1,os.path.dirname(SCRIPT_DIR))
from Moska import Deck

class TestDeck(unittest.TestCase):
    deck = None
    card = None
    def setUp(self) -> None:
        self.deck = Deck.StandardDeck(shuffle = True)
        self.card = self.deck.pop_cards(1)[0]
        
    def test_place_to_bottom(self):
        card = copy.copy(self.card)
        self.deck.place_to_bottom(self.card)
        # Pop from right where the cards just placed to
        new_card = self.deck.cards.pop()
        self.assertTrue(card == new_card)
        
    def test_len52(self):
        self.assertTrue(len(self.deck) == 51)
        
if __name__ == "__main__":
    unittest.main()