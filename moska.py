import logging
import os
import time
from Moska.Game import MoskaGame
from Moska.Player.AbstractPlayer import AbstractPlayer
from Moska.Player.HumanPlayer import HumanPlayer
import multiprocessing
from typing import List

def start_threaded_moska(players : List[AbstractPlayer],file = "",timeout=2, random_seed = None):
    moskaGame = MoskaGame(players=players,
                          log_file=file,
                          log_level=logging.DEBUG,
                          timeout=timeout,
                          random_seed=random_seed,
                          
                          )
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
    moskaGame = MoskaGame(players = players,log_level=logging.DEBUG,timeout=120)
    moskaGame.start()

def play_games(n=1,nplayers=5,log_prefix="moskafile_",cpus=-1, chunksize=-1):
    print(f"Found CPUs: {os.cpu_count()}")
    try:
        avail_cpus = len(os.sched_getaffinity(0))
        print(f"CPUs available for use: {avail_cpus}")
    except AttributeError as ae:
        print(f"No information on CPU availability, assuming N - 1")
        avail_cpus = os.cpu_count() - 1
        pass
    start_time = time.time()
    cpus = min(avail_cpus,n) if cpus==-1 else cpus
    chunksize = n//cpus if chunksize == -1 else chunksize
    print(f"Starting a pool with {cpus} processes and {chunksize} chunksize...")
    arg_list = []
    for i in range(n):
        file = log_prefix + f"({i}).log"    # Name for the Games log file
        players = MoskaGame._get_random_players(nplayers)   # n random players
        for pl in players:
            pl.log_file = f"{pl.name}_({i}).log"            # Set each players log files
            pl.log_level = logging.DEBUG
        arg_list.append((players, file))
        #processes.append(multiprocessing.Process(target=start_threaded_moska,args=(players,file)))
    results = []
    with multiprocessing.Pool(cpus) as pool:
        print("Games running...")
        gen = pool.imap_unordered(start_threaded_moska_process,arg_list,chunksize = chunksize)
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
    print(f"Simulated {len(results)} games. {len(results) - failed_games} succesful games. {failed_games} failed.")
    print(f"Time taken: {time.time() - start_time}")
    ranks = {}
    for res in results:
        if res is None:
            continue
        lastid = res[-1][0].split("-")[0]
        if lastid not in ranks:
            ranks[lastid] = 0
        ranks[lastid] += 1
    rank_list = list(ranks.items())
    rank_list.sort(key=lambda x : x[1])
    for pl,rank in rank_list:
        print(f"{pl} was last {round(100*rank/(len(results)-failed_games),2)} % times")


if __name__ == "__main__":
    n = 5
    if not os.path.isdir("Logs"):
        os.mkdir("Logs")
    os.chdir("Logs/")
    #play_as_human(n)
    play_games(100,nplayers=5,log_prefix="moskafile_")
    
    

