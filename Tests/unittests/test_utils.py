import unittest
import sys
import os
# Add the main directory to the path
sys.path.append(os.path.abspath(".\\"))
#from Moska import Hand,Game,Deck,utils, Player
from Moska.Game.Game import MoskaGame
from Moska.Game.Hand import MoskaHand
from Moska.Game.Deck import StandardDeck, Card
from Moska.Game.utils import check_can_fall_card
from Moska.Player.MoskaBot0 import MoskaBot0

class TestUtils(unittest.TestCase):
    game = None
    player = None
    deck = None
    def setUp(self):
        """For some reason this doesn't appear to be called before each 'test_'' method
        """
        self.deck = StandardDeck()
        players = [MoskaBot0(name="Bot0"),MoskaBot0(name="Bot1"),MoskaBot0(name="Bot2"),MoskaBot0(name="Bot3"),MoskaBot0(name="Bot4"),MoskaBot0(name="Bot5")]
        self.game = MoskaGame(players=players)
        self.player = self.game.players[0]
        self.game.turnCycle.ptr = 0
    
    def tearDown(self) -> None:
        #del self.deck, self.game.turnCycle, self.game, self.player, self.game
        return super().tearDown()
    
    def test_can_fall(self):
        cards = []
        trump = "H"
        cards.append(Card(2,"H"))
        cards.append(Card(6,"S"))
        table_cards = []
        table_cards.append(Card(14,"H"))
        table_cards.append(Card(5,"S"))
        for card in cards:
            for tcard in table_cards:
                success = check_can_fall_card(card,tcard,trump)
                if card.value == 2 and card.suit == "H":
                    self.assertTrue(not success or tcard.suit == "S")
                if card.value == 6:
                    self.assertTrue(not success or tcard.value == 5)
                    
    
    def test_TurnCycle_starts_at_0(self):
        tc = self.game.turnCycle
        self.assertTrue(tc.get_at_index() is self.player)
        self.assertTrue(tc.ptr == 0)
    
    def test_TurnCycle_get_next(self):
        tc = self.game.turnCycle
        nxt = tc.get_at_index(1)
        self.assertTrue(nxt is tc.get_next(incr_ptr=False))
        
    def test_TurnCycle_get_prev(self):
        tc = self.game.turnCycle
        prev = tc.get_at_index(tc.ptr - 1)
        self.assertTrue(prev is tc.get_prev(incr_ptr=False))
        
    def test_TurnCycle_increments_ptr(self):
        tc = self.game.turnCycle
        print(f"ptr: {tc.ptr}")
        tc.get_next_condition(lambda x : x.pid == 2)
        #ptr should be 2
        print(f"ptr: {tc.ptr}")
        self.assertTrue(tc.ptr == 2)
        tc.ptr = 0  # Because setUp doesn't reset ptr :d
        
    def test_TurnCycle_doesnt_infiniteloop(self):
        tc = self.game.turnCycle
        out = tc.get_next_condition(cond = lambda x : x.pid == 68)
        self.assertTrue(not out)
        tc.ptr = 0
    
                
                
      
if __name__ == "__main__":
    unittest.main()