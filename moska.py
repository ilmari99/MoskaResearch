import Deck
from Game import MoskaGame
from Player import MoskaPlayerBase
import sys

if __name__ == "__main__":
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
