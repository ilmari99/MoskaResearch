import logging
import os
from time import time
from Moska import Deck
from Moska.Game import MoskaGame
from Moska.Player import HumanPlayer, MoskaBot1, MoskaPlayer
import sys
import multiprocessing
import random
    
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
    for i in range(5):
        if random.random() > 0.5:
            players.append(MoskaPlayer(moskaGame,pid=i,debug=True,log_file=f"P{i}_"+file+".log"))
        else:
            players.append(MoskaBot1(moskaGame,pid=i,debug=True,log_file=f"P{i}_"+file+".log"))
    for player in players:
        moskaGame.add_player(player)
    ranks = moskaGame.start()
    sys.stdout = sys.__stdout__
    print(ranks[0])
    return True

def play_as_human(nopponents):
    deck = Deck.StandardDeck()
    moskaGame = MoskaGame(deck)
    moskaGame.add_player(HumanPlayer(moskaGame,pid = 101010, name = "human", debug=True,log_file="human.log",log_level = logging.DEBUG))
    for i in range(nopponents):
        player = MoskaPlayer(moskaGame,pid=i,debug=True,delay=1)
        moskaGame.add_player(player)
    moskaGame.start()
    
def play_games(n=1):
    processes = []
    bfile = "moskafile_threaded"
    for i in range(n):
        file = bfile + f"({i}).txt"
        processes.append(multiprocessing.Process(target=start_threaded_moska,args=(file,)))
    for prc in processes:
        prc.start()
    print("Games running...")
    timeout = 3
    st = time()
    while time() - st <timeout:
        if not any((p.is_alive() for p in processes)):
            break
    if any((p.is_alive() for p in processes)):
        print("Some processes timedout.")
        for prc in processes:
            prc.terminate()
            prc.join()
    print("Games finished")
    

if __name__ == "__main__":
    #n = 5
    play_as_human(4)
    #play_as_human(4)
    
    
