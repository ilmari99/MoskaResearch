import unittest
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/")
SCRIPT_DIR = "\\".join(SCRIPT_DIR[0:-2])
#print(SCRIPT_DIR)
sys.path.insert(1,os.path.dirname(SCRIPT_DIR))
from Moska import Hand,Game,Deck,utils, Player

class TestUtils(unittest.TestCase):
    game = None
    player = None
    deck = None
    def setUp(self):
        """For some reason this doesn't appear to be called before each 'test_'' method
        """
        self.deck = Deck.StandardDeck()
        self.game = Game.MoskaGame(self.deck)
        self.player = Player.MoskaPlayer(self.game,pid=1010,name="test")
        if self.game.players:
            self.player = self.game.players[0]
            return
        self.game.add_player(self.player)
        for p in [Player.MoskaPlayer(self.game,pid=i) for i in range(3)]:
            print([p.pid for p in self.game.turnCycle.population])
            self.game.add_player(p)
    
    def tearDown(self) -> None:
        #del self.deck, self.game.turnCycle, self.game, self.player, self.game
        return super().tearDown()
    
    def test_can_fall(self):
        cards = []
        triumph = "H"
        cards.append(Deck.Card(2,"H"))
        cards.append(Deck.Card(6,"S"))
        table_cards = []
        table_cards.append(Deck.Card(14,"H"))
        table_cards.append(Deck.Card(5,"S"))
        for card in cards:
            for tcard in table_cards:
                success = utils.check_can_fall_card(card,tcard,triumph)
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
        tc.get_next_condition(lambda x : x.pid == 1)
        #ptr should be 2
        self.assertTrue(tc.ptr == 2)
        tc.ptr = 0  # Because setUp doesn't reset ptr :d
        
    def test_TurnCycle_doesnt_infiniteloop(self):
        tc = self.game.turnCycle
        out = tc.get_next_condition(cond = lambda x : x.pid == 68)
        self.assertTrue(not out)
        tc.ptr = 0
    
                
                
      
if __name__ == "__main__":
    unittest.main()