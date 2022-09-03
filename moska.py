import Deck
from Game import MoskaGame, MoskaGameThreaded
from Player import MoskaPlayerBase, MoskaPlayerThreadedBase
import sys

def start_moska():
    moskafile = open("./moskafile.txt","w",encoding="utf-8")
    sys.stdout = moskafile
    deck = Deck.StandardDeck()
    moskaGame = MoskaGame(deck)
    players = []
    for i in range(5):
        players.append(MoskaPlayerBase(moskaGame,pid=i))
    for player in players:
        moskaGame.add_player(player)
    moskaGame.start()
    moskafile.close()
    
def start_threaded_moska():
    moskafile = open("./moskafile_threaded.txt","w",encoding="utf-8")
    sys.stdout = moskafile
    deck = Deck.StandardDeck()
    moskaGame = MoskaGameThreaded(deck)
    players = []
    for i in range(5):
        players.append(MoskaPlayerThreadedBase(moskaGame,pid=i))
    for player in players:
        moskaGame.add_player(player)
    moskaGame.start()
    

if __name__ == "__main__":
    start_threaded_moska()
