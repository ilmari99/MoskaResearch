#!/usr/bin/env python3
import logging
import os
import time
import sys
from Moska.Game.Game import MoskaGame
from Moska.Player.MoskaBot3 import MoskaBot3
from Moska.Player.AbstractPlayer import AbstractPlayer
from Moska.Player.HumanPlayer import HumanPlayer
import multiprocessing
from typing import Any, Callable, Dict, Iterable, List, Tuple
from Moska.Player.MoskaBot2 import MoskaBot2
from Moska.Player.MoskaBot0 import MoskaBot0
from Moska.Player.MoskaBot1 import MoskaBot1
from Moska.Player.RandomPlayer import RandomPlayer
from Moska.Player.NNEvaluatorBot import NNEvaluatorBot
from Moska.Player.NNHIFEvaluatorBot import NNHIFEvaluatorBot
from Moska.Player.HeuristicEvaluatorBot import HeuristicEvaluatorBot
from Moska.Player.ModelBot import ModelBot
import random
import numpy as np
from scipy.optimize import minimize
from scipy import stats as sc_stats
#from noisyopt import minimizeCompass,minimize



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


def args_to_gamekwargs(
    game_kwargs : Callable,
    players : List[Tuple[AbstractPlayer,Callable]],
    gameid : int,
    shuffle : False,
                        ) -> Dict[str,Any]:
    """Turn a dynamic arguments (for ex callable changing game log), to a static gamekwargs dictionary.

    Args:
        game_kwargs (Callable): _description_
        players (List[Tuple[AbstractPlayer,Callable]]): _description_
        gameid (int): _description_
        shuffle (False): _description_

    Returns:
        _type_: _description_
    """
    game_args = game_kwargs(gameid)
    players = [pl(**args(gameid)) for pl, args in players]
    if not players:
        assert "nplayers" in game_args or "players" in game_args
    else:
        game_args["players"] = players
    if shuffle and "players" in game_args:
        random.shuffle(game_args["players"])
    return game_args
    
def play_as_human(game_id = 0):
    shared_kwargs = {
        "log_level" : logging.DEBUG,
        "delay":1,
    }
    players = [
        (HumanPlayer,lambda x : {"name":"Human","log_file":f"human-{x}.log","requires_graphic":True}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV1","requires_graphic":True,
                                                  "log_file":f"Game-{x}-NNEV1.log", 
                                                  "max_num_states":8000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV2","requires_graphic":True,
                                                  "log_file":f"Game-{x}-NNEV2.log", 
                                                  "max_num_states":8000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV3","requires_graphic":True,
                                                  "log_file":f"Game-{x}-NNEV3.log", 
                                                  "max_num_states":8000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
    ]
    gamekwargs = lambda x : {
        "log_file" : f"Humangame-{x}.log",
        "players" : players,
        "log_level" : logging.DEBUG,
        "timeout" : 1000,
        "model_paths":["../Models/ModelMB11-260/model.tflite"]
    }
    game = args_to_gamekwargs(gamekwargs,players,gameid = game_id,shuffle = True)
    game = MoskaGame(**game)
    return game.start()

def run_game(kwargs):
    return MoskaGame(**kwargs).start()

def play_games(players : List[Tuple[AbstractPlayer,Callable]],
               game_kwargs : Callable,
               n : int = 1,
               cpus :int = -1,
               chunksize : int = -1,
               shuffle_player_order : bool = True,
               verbose : bool = True,
               ):
    """ Simulate moska games with specified players. Return the rankings in finishing order.
    The players are specified by a list of tuples, with AbstractPlayer subclass and argument pairs.

    Args:
        players (List[Tuple[AbstractPlayer,Callable]]): The players are specified by a list of tuples, with (AbstractPlayer subclass, Callable -> dict) pairs.
        game_kwargs (Callable): A callable, that takes in the gameid, and returns the desired game arguments
        n (int, optional): Number of games to play. Defaults to 1.
        cpus (int, optional): Number of processes to start simultaneously. Defaults to the number of cpus.
        chunksize (int, optional): How many games to initially give each process. Defaults to defaults to n // cpus.
        shuffle_player_order (bool, optional) : Whether to randomly shuffle the player order in the game.

    Returns:
        Dict: _description_
    """
    
    start_time = time.time()
    # Select the specified number of cpus, or how many cpus are available
    cpus = min(os.cpu_count(),n) if cpus==-1 else cpus
    # Select the chunksize, so that it is close to 'chunksize * cpus = ngames'
    chunksize = n//cpus if chunksize == -1 else chunksize
    
    arg_gen = (args_to_gamekwargs(game_kwargs,players,i,shuffle_player_order) for i in range(n))
    results = []
    print(f"Starting a pool with {cpus} processes and {chunksize} chunksize...")
    with multiprocessing.Pool(cpus) as pool:
        # Lazily run games distributing 'chunksize' games to a process. The results will not be ordered.
        gen = pool.imap_unordered(run_game,arg_gen,chunksize = chunksize)
        failed_games = 0
        start = time.time()
        # Loop while there are games.
        while gen:
            # Print statistics every 10 seconds. TODO: Not working, fix.
            if verbose and int(time.time() - start) % 10 == 0:
                print(f"Simulated {len(results)/n*100:.2f}% of games. {len(results) - failed_games} succesful games. {failed_games} failed.")
            try:
                # res contains either the finish ranks, or None if the game failed
                res = next(gen)
            except StopIteration as si:
                break
            if res is None:
                failed_games += 1
            results.append(res)
    print(f"Simulated {len(results)/n * 100:.2f}% of games. {len(results) - failed_games} succesful games. {failed_games} failed.")
    print(f"Time taken: {time.time() - start_time}")
    return results

def get_loss_percents(results, player="all", show = True):
    # Return the results as a dictionary of player names and their loss percentage
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

def to_minimize_func(params):
    """ The function to minimize. You must change the parameters here."""
    # Mapping of the coefficients to the names of the parameters
    params = {
        "my_cards" : params[0],
        "len_set_my_cards" : params[1],
        "len_my_cards" : params[2],
        "kopled":params[3],
        "missing_card" : params[4],
    }
    shared_kwargs = {
        "log_level" : logging.WARNING,
    }
    print(f"Simulating with params: {params}")
    players = [
        #(ModelBot, lambda x : {**shared_kwargs, **{"name" : f"MB-1","log_file":f"Game-{x}-MB-1.log","max_num_states":1000,
        #                                          "state_prediction_format":"FullGameState-old", 
        #                                          "normalize_state_vector" : False}}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV","log_file":f"Game-{x}-HEV.log", "max_num_states":1000, "coefficients" : params}}),
        (MoskaBot2, lambda x : {**shared_kwargs,**{"name" : f"B2-1","log_file":f"Game-{x}-B-3.log"}}),# "model_file":"/home/ilmari/python/moska/Model5-300/model.tflite", "requires_graphic" : False}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV",
                                                  "log_file":f"Game-{x}-NNEV.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-2","log_file":f"Game-{x}-B-4.log"}}),
    ]
    gamekwargs = lambda x : {
            "log_file" : f"Game-{x}.log",
            "log_level" : logging.WARNING,
            "timeout" : 30,
            "gather_data":True,
            "model_paths":["../Models/ModelMB10-160/model.tflite"],
        }
    player_to_minimize = "HEV"
    results = play_games(players, gamekwargs, n=500, cpus=10, chunksize=2,shuffle_player_order=True, verbose=False)
    out = get_loss_percents(results,player=player_to_minimize, show=False)
    print(f"Player '{player_to_minimize}' lost: {out} %")
    return out

def to_minimize_call():
    x0 = [1, 1, -1, 1, 51]
    res = minimize(to_minimize_func,x0=x0,method="Powell",options={"maxiter":600,"disp":True})
    print(f"Minimization result: {res}")
    return




if __name__ == "__main__":
    new_dir = "Logs"
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    os.chdir(new_dir + "/")
    if not os.path.isdir("Vectors"):
        os.mkdir("Vectors")

    shared_kwargs = {
        "log_level" : logging.WARNING,
        "delay":0,
    }

    # The {**dict1, **dict2} operation is used to merge dictionaries, with the latter overwriting the former for shared keys
    much_players = [
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV","log_file":f"Game-{x}-HEV.log", "max_num_states":10000}}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV-small","log_file":f"Game-{x}-HEV-small.log", "max_num_states":100}}),
        #(NNHIFEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNHIFEV",
        #                                        "log_file":f"Game-{x}-NNHIFEV.log",
        #                                        "max_num_states":10000,
        #                                        "pred_format":"new"
        #                                        }}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV-old",
                                            "log_file":f"Game-{x}-NNEV-old.log", 
                                            "max_num_states":10000,
                                            "pred_format":"old",
                                            "model_id":0,
                                            }}),
        (MoskaBot2, lambda x : {**shared_kwargs,**{"name" : f"B2-1","log_file":f"Game-{x}-B-3.log"}}),# "model_file":"/home/ilmari/python/moska/Model5-300/model.tflite", "requires_graphic" : False}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV-new",
                                                  "log_file":f"Game-{x}-NNEV-new.log", 
                                                  "max_num_states":10000,
                                                  "pred_format":"new",
                                                  "model_id":1,
                                                  }}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-2","log_file":f"Game-{x}-B-4.log"}}),
    ]

    gamekwargs = lambda x : {
        "log_file" : f"Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 30,
        "gather_data":True,
        "model_paths":["../Models/ModelMB11-260/model.tflite", "../Models/ModelF4/model.tflite"]
    }
    for i in range(1):
        players = much_players[2:]
        results = play_games(players, gamekwargs, n=100, cpus=10, chunksize=3,shuffle_player_order=True,verbose=False)
        #results = play_games(act_players, gamekwargs, n=1000, cpus=10, chunksize=10,shuffle_player_order=True)
        res = get_loss_percents(results,player="all", show=True)
        print(res)
        
        
        