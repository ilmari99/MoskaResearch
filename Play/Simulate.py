#!/usr/bin/env python3
import logging
import os
import sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import sys
from Moska.Game.Game import MoskaGame
from Moska.Player.AbstractPlayer import AbstractPlayer
import multiprocessing
from typing import Any, Callable, Dict, Iterable, List, Tuple
from Utils import make_log_dir, args_to_gamekwargs
from PlayerWrapper import PlayerWrapper

"""This file contains simulation utility functions for playing (multiple) games of Moska."""

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


def run_game(kwargs):
    """ Run a single game with the specified arguments. Return the finishing ranks, or None if the game failed. 
    """
    return MoskaGame(**kwargs).start()

def play_games(players : List[PlayerWrapper],
               game_kwargs : Callable,
               ngames : int = 1,
               cpus :int = -1,
               chunksize : int = 1,
               shuffle_player_order : bool = True,
               verbose : bool = True,
               ):
    """ Simulate multiple moska games with specified players. Return the rankings in finishing order.
    The players are specified by a list of tuples, with AbstractPlayer subclass and argument pairs.

    Args:
        players (List[Tuple[AbstractPlayer,Callable]]): The players are specified by a list of tuples, with (AbstractPlayer subclass, Callable -> dict) pairs.
        game_kwargs (Callable): A callable, that takes in the gameid, and returns the desired game arguments
        ngames (int, optional): Number of games to play. Defaults to 1.
        cpus (int, optional): Number of processes to start simultaneously. Defaults to -1 = the number of cpus.
        chunksize (int, optional): How many games to initially give each process. Defaults to 1
        shuffle_player_order (bool, optional) : Whether to randomly shuffle the player order in the game.

    Returns:
        list[List] : A list of lists, where each sublist contains the finishing ranks of a game.
    """
    
    start_time = time.time()
    # Select the specified number of cpus, or how many cpus are available
    cpus = min(os.cpu_count(),ngames) if cpus==-1 else cpus
    
    arg_gen = (args_to_gamekwargs(game_kwargs,players,i,shuffle_player_order) for i in range(ngames))
    results = []
    print(f"Starting a pool with {cpus} processes and {chunksize} chunksize...")
    with multiprocessing.Pool(cpus) as pool:
        # Lazily run games distributing 'chunksize' games to each process. The results will not be ordered.
        gen = pool.imap_unordered(run_game,arg_gen,chunksize = chunksize)
        failed_games = 0
        start = time.time()
        # Loop while there are games.
        while gen:
            # Print statistics every 10 seconds. TODO: Not working, fix.
            if verbose and int(time.time() - start) % 10 == 0:
                print(f"Simulated {len(results)/ngames*100:.2f}% of games. {len(results) - failed_games} succesful games. {failed_games} failed.",flush=True)
            try:
                # res contains either the finish ranks, or None if the game failed
                res = next(gen)
            except StopIteration as si:
                break
            if res is None:
                failed_games += 1
            results.append(res)
    
    print(f"Simulated {len(results)/ngames * 100:.2f}% of games. {len(results) - failed_games} succesful games. {failed_games} failed.")
    print(f"Time taken: {time.time() - start_time}")
    if verbose:
        get_loss_percents(results)
    return results

def get_loss_percents(results, player="all", show = True):
    """ Return and/or print the results as a dictionary of player names and their loss percentage.
    Doesnt print the player if they have no losses.
    """
    losses = {}
    games = 0
    results_filtered = filter(lambda x : x is not None,results)
    # A single results is represented as a List[Tuple[playername,rank]]
    for res in results_filtered:
        games += 1
        lastid = res[-1][0]
        #lastid = res[-1][0].split("-")[0]
        if lastid not in losses:
            losses[lastid] = 0
        losses[lastid] += 1
    loss_list = list(losses.items())
    # Sort the losses by the number of losses
    loss_list.sort(key=lambda x : x[1])
    # Return the losses as a dictionary. The dictionary is ordered
    loss_percentages = {k : round((v/games)*100,4) for k,v in loss_list}
    if show:
        for pl,loss_perc in loss_percentages.items():
            print(f"{pl} was last {loss_perc} % times")
    # return the full dictionary if player="all"
    if player=="all":
        return loss_percentages
    else:
        return loss_percentages[player] if player in loss_percentages else 0