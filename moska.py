import logging
import os
import time
from Moska.Game import MoskaGame
from Moska.Player.AbstractPlayer import AbstractPlayer
from Moska.Player.HumanPlayer import HumanPlayer
import multiprocessing
from typing import Any, Callable, Dict, Iterable, List, Tuple
from Moska.Player.MoskaBot2 import MoskaBot2
from Moska.utils import add_before

def set_game_args(game : MoskaGame, gamekwargs : Dict[str,Any]) -> None:
    """Sets a game instances variables from a dictionary of key-value pairs.
    If value is Callable, the returned value is assigned.

    Args:
        game (MoskaGame): The game whose attributes to set
        gamekwargs (Dict): _description_
    """
    for k,v in gamekwargs.items():
        if isinstance(v,Callable):
            v = v(game)
        game.__setattr__(k,v)
    return

def set_player_args(players : Iterable[AbstractPlayer], plkwargs : Dict[str,Any]) -> None:
    """Sets a player instances variables from a dictionary of key-value pairs.
    'players' must be an iterable, all of whose attributes are set to values found in 'plkwargs'
    If a value is Callable, the returned value is assigned.

    Args:
        players (Iterable[AbstractPlayer]): The players whose attributes will be set
        plkwargs (Dict[str,Any]): The attributes and corresponding values
    """
    for pl in players:
        for k,v in plkwargs.items():
            if isinstance(v,Callable):
                v = v(pl)
            pl.__setattr__(k,v)
    return


def start_moska_process(
                        gamekwargs : Dict[str,Any] = {},
                        plkwargs : Dict[str,Any] = {},
                        gameid : int = 0,
                        ):
    moskaGame = MoskaGame()
    set_game_args(moskaGame,gamekwargs)
    set_player_args(moskaGame.players,plkwargs)
    for pl in moskaGame.players:
        pl.log_file = add_before(".",pl.name + ".log","("+str(gameid)+")")
    return moskaGame.start()

def start_moska_process_wrap(args : Tuple):
    return start_moska_process(*args)
    
def play_as_human(nopponents):
    players = [HumanPlayer(name="Human-",log_file="human.log")] + MoskaGame._get_random_players(nopponents)
    player_kwargs = {
        "delay":1,
        "log_level":logging.DEBUG,
    }
    gamekwargs = {
        "log_file" : "Humangame.log",
        "players" : players,
        "log_level" : logging.DEBUG,
        "timeout" : 120,
    }
    return start_moska_process(gamekwargs=gamekwargs,plkwargs=player_kwargs)

def play_games(n=1,nplayers=5,log_prefix="moskafile",cpus=-1, chunksize=-1):
    start_time = time.time()
    avail_cpus = os.cpu_count()
    cpus = min(avail_cpus,n) if cpus==-1 else cpus
    chunksize = n//cpus if chunksize == -1 else chunksize
    game_kwargs = lambda p : {
        "nplayers" : nplayers,
        "log_file" : log_prefix + "(" +str(p)+ ")" + ".log",
        "log_level" : logging.DEBUG,
        "timeout" : 3,
    }
    player_kwargs = {
        "log_level": logging.DEBUG,
    }
    arg_gen = ((game_kwargs(i),player_kwargs,i) for i in range(n))
    results = []
    print(f"Starting a pool with {cpus} processes and {chunksize} chunksize...")
    with multiprocessing.Pool(cpus) as pool:
        print("Games running...")
        gen = pool.imap_unordered(start_moska_process_wrap,arg_gen,chunksize = chunksize)
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
    play_games(10,nplayers=5,log_prefix="moskafile",cpus=4,chunksize=1)
    
    

