import logging
import os
from time import time
from Moska.Game import MoskaGame
from Moska.Player.BasePlayer import BasePlayer
from Moska.Player.HumanPlayer import HumanPlayer
import multiprocessing
from typing import List

def start_threaded_moska(players : List[BasePlayer],file = "",timeout=10, random_seed = None):
    moskaGame = MoskaGame(players=players,log_file=file,log_level=logging.DEBUG,timeout=timeout, random_seed=random_seed)
    ranks = moskaGame.start()
    return ranks

def start_threaded_moska_process(t : tuple):
    return start_threaded_moska(*t)
    
def play_as_human(nopponents):
    players = [HumanPlayer(pid=69,log_file="human.log")] + MoskaGame._get_random_players(nopponents)
    for i,pl in enumerate(players):
        pl.delay = 1
        pl.log_file = f"{pl.name}_({i}).log"
        pl.log_level = logging.DEBUG
    moskaGame = MoskaGame(players = players,log_level=logging.DEBUG)
    moskaGame.start()

def play_games(n=1,nplayers=5,log_prefix="moskafile_"):
    pool = multiprocessing.Pool(n)
    timeout = 3
    arg_list = []
    print(f"Available CPUs: {os.cpu_count()}")
    for i in range(n):
        file = log_prefix + f"({i}).log"    # Name for the Games log file
        players = MoskaGame._get_random_players(nplayers)   # n random players
        for pl in players:
            pl.log_file = f"{pl.name}_({i}).log"            # Set each players log files
            pl.log_level = logging.DEBUG
        arg_list.append((players, file))
        #processes.append(multiprocessing.Process(target=start_threaded_moska,args=(players,file)))
    results = []
    with multiprocessing.Pool(os.cpu_count()-1) as pool:
        print("Games running...")
        gen = pool.imap_unordered(start_threaded_moska_process,arg_list,chunksize = os.cpu_count() - 1)
        print(gen)
        failed_games = 0
        while gen:
            try:
                res = next(gen)
            except StopIteration as si:
                break
            if res is None:
                failed_games += 1
                res = None
            print(res)
            results.append(res)
    print(f"Simulated {len(results)} games. {len(results) - failed_games} succesfull games. {failed_games} failed.")
    print("Results: ", results)
    b1_last = 0
    b0_last = 0
    for res in results:
        if res is None:
            continue
        if "B1" in res[-1][0]:
            b1_last += 1
        elif "B0" in res[-1][0]:
            b0_last += 1
    print(f"B1 was last {b1_last} times")
    print(f"B0 was last {b0_last} times")


if __name__ == "__main__":
    n = 5
    if not os.path.isdir("Logs"):
        os.mkdir("Logs")
    os.chdir("Logs/")
    #play_as_human(n)
    play_games(500,nplayers=5,log_prefix="moskafile_")
    
    

