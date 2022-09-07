from Moska import Deck
from Moska.Game import MoskaGame
from Moska.Player import HumanPlayer, MoskaPlayer
import sys
import multiprocessing

    
def start_threaded_moska(file = "moskafile_threaded.txt"):
    fsplit = file.split(".")
    file = fsplit[0]
    ftype = fsplit[-1]
    stdout = open(file + "."+ftype,"w",encoding="utf-8")
    stderr = open(file + "_stderr"+"."+ftype, "w", encoding="utf-8")
    sys.stdout = stdout
    sys.stderr = stderr
    deck = Deck.StandardDeck()
    moskaGame = MoskaGame(deck)
    players = []
    #moskaGame.add_player(HumanPlayer(moskaGame,pid=11,name = "ilmari"))
    for i in range(5):
        players.append(MoskaPlayer(moskaGame,pid=i))
    for player in players:
        moskaGame.add_player(player)
    moskaGame.start()
    return True
    
def play_games(n=1):
    processes = []
    bfile = "moskafile_threaded"
    for i in range(n):
        file = bfile + f"({i}).txt"
        processes.append(multiprocessing.Process(target=start_threaded_moska,args=(file,)))
    for prc in processes:
        prc.start()
    print("Games running...")
    for prc in processes:
        prc.join()
    print("Games finished")
    

if __name__ == "__main__":
    #n = 5
    play_games(1)
    
    
