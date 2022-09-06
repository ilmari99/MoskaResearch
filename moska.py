from concurrent.futures import process
import Deck
from Game import MoskaGame, MoskaGameThreaded
from Player import MoskaPlayerBase, MoskaPlayerThreadedBase
import sys
import multiprocessing

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
    
def start_threaded_moska(file = "moskafile_threaded.txt"):
    fsplit = file.split(".")
    file = fsplit[0]
    ftype = fsplit[-1]
    #print(file)
    #print(ftype)
    stdout = open(file + "."+ftype,"w",encoding="utf-8")
    stderr = open(file + "_stderr"+"."+ftype, "w", encoding="utf-8")
    sys.stdout = stdout
    sys.stderr = stderr
    deck = Deck.StandardDeck()
    moskaGame = MoskaGameThreaded(deck)
    players = []
    for i in range(5):
        players.append(MoskaPlayerThreadedBase(moskaGame,pid=i))
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
    n = 5
    play_games(7)
    
    
