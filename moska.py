import logging
import os
from time import time
from Moska import Deck
from Moska.Game import MoskaGame
from Moska.BasePlayer import BasePlayer
from Moska.Player import HumanPlayer, MoskaBot1
import sys
import multiprocessing
import random
from typing import TextIO, List

def start_threaded_moska(players : List[BasePlayer],file = ""):
    moskaGame = MoskaGame(players=players,log_file=file,log_level=logging.DEBUG)
    moskaGame.start()
    
def play_as_human(nopponents):
    players = [HumanPlayer(pid=69,log_file="human.log")] + MoskaGame._get_random_players(nopponents)
    for i,pl in enumerate(players):
        pl.delay = 1
        pl.log_file = f"{pl.name}_({i}).log"
        pl.log_level = logging.DEBUG
    moskaGame = MoskaGame(players = players,log_level=logging.DEBUG)
    moskaGame.start()

def play_games(n=1,nplayers=5,log_prefix="moskafile_"):
    processes = []
    for i in range(n):
        file = log_prefix + f"({i}).log"    # Name for the Games log file
        players = MoskaGame._get_random_players(nplayers)   # n random players
        for pl in players:
            pl.log_file = f"{pl.name}_({i}).log"            # Set each players log files
            pl.log_level = logging.DEBUG
        processes.append(multiprocessing.Process(target=start_threaded_moska,args=(players,file)))
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
    n = 5
    if not os.path.isdir("Logs"):
        os.mkdir("Logs")
    os.chdir("Logs/")
    #play_as_human(n)
    play_games(1,nplayers=5,log_prefix="moskafile_")
    
    

