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

def set_player_args_optimize_bot3(players : Iterable[AbstractPlayer], plkwargs : Dict[str,Any],coeffs = {}):
    for pl in players:
        for k,v in plkwargs.items():
            if k == "coefficients" and not isinstance(pl,MoskaBot3):
                continue
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
        "delay":0,
    }
    players = [
        (HumanPlayer,lambda x : {"name":"Human","log_file":f"human-{x}.log","requires_graphic":True}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV1",
                                                  "log_file":f"Game-{x}-NNEV1.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV2",
                                                  "log_file":f"Game-{x}-NNEV2.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV3",
                                                  "log_file":f"Game-{x}-NNEV3.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
    ]
    gamekwargs = lambda x : {
        "log_file" : f"Humangame-{x}.log",
        "players" : players,
        "log_level" : logging.DEBUG,
        "timeout" : 1000,
        "model_paths":["../Models/ModelMB8-100/model.tflite"]
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
            if int(time.time() - start) % 10 == 0:
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
    "PlayFallFromDeck" : params[0],
    "PlayFallFromHand" : params[1],
    "PlayToSelf" : params[2],
    "InitialPlay" : params[3],
    "Skip" : params[4],
    "EndTurn" : params[5],
    "PlayToOther" : params[6]
    }
    print("coeffs",params)
    shared_kwargs = {
        "log_level" : logging.WARNING,
    }
    players = [
            (ModelBot, lambda x : shared_kwargs | {"name" : f"MB-1","log_file":f"Game-{x}-MB-1.log","max_num_states":100, "parameters":params}),
            (MoskaBot3, lambda x : shared_kwargs | {"name" : f"B3-1","log_file":f"Game-{x}-B-2.log"}),
            (MoskaBot2, lambda x : shared_kwargs |  {"name" : f"B2-1","log_file":f"Game-{x}-B-3.log"}),
            (MoskaBot2, lambda x : shared_kwargs | {"name" : f"B2-2","log_file":f"Game-{x}-MB-4.log"}),
            ]
    gamekwargs = lambda x : {
            "log_file" : f"Game-{x}.log",
            "log_level" : logging.WARNING,
            "timeout" : 15,
        }
    player_to_minimize = "MB1-1"
    results = play_games(players, gamekwargs, n=600, cpus=10, chunksize=6,disable_logging=False)
    out = get_loss_percents(results,player=player_to_minimize, show=False)
    print(f"Player {player_to_minimize} lost: {out} %")
    return out

def to_minimize_call():
    x0 = [1 for _ in range(7)]
    bounds = [(0,1) for _ in range(7)]
    res = minimize(to_minimize_func,x0=x0,method="Nelder-Mead",bounds=bounds)
    print(f"Minimization result: {res}")
    return


if __name__ == "__main__":
    new_dir = "TestLog"
    
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    os.chdir(new_dir + "/")
    if not os.path.isdir("Vectors"):
        os.mkdir("Vectors")
    #for i in range(1,10):
    #    play_as_human(game_id=i)
    #exit()

    shared_kwargs = {
        "log_level" : logging.DEBUG,
        "delay":0,
    }

    # The {**dict1, **dict2} operation is used to merge dictionaries, with the latter overwriting the former for shared keys
    players = [
        #(ModelBot, lambda x : {**shared_kwargs, **{"name" : f"MB-1","log_file":f"Game-{x}-MB-1.log","max_num_states":1000,
        #                                          "state_prediction_format":"FullGameState-old", 
        #                                          "normalize_state_vector" : False}}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV","log_file":f"Game-{x}-HEV.log"}}),
        (MoskaBot2, lambda x : {**shared_kwargs,**{"name" : f"B2-1","log_file":f"Game-{x}-B-3.log"}}),# "model_file":"/home/ilmari/python/moska/Model5-300/model.tflite", "requires_graphic" : False}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV",
                                                  "log_file":f"Game-{x}-NNEV.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False}}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-2","log_file":f"Game-{x}-B-4.log","parameters" : {'fall_card_already_played_value': -0.5, 
         'fall_card_same_value_already_in_hand': 0.2, 
         'fall_card_card_is_preventing_kopling': -0.4, 
         'fall_card_deck_card_not_played_to_unique': 0.2, 
         'fall_card_threshold_at_start': 15.0, 
         'initial_play_quadratic_scaler': 0.7
         }}}),
    ]
    
    __players__ = [
        (MoskaBot3,lambda x : {"name":"B3","log_file":f"Game-{x}-B3.log"}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV1",
                                                  "log_file":f"Game-{x}-NNEV1.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False,
                                                  "model_id":0
                                                  }}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV2",
                                                  "log_file":f"Game-{x}-NNEV2.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False,
                                                  "model_id":1
                                                  }}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV3",
                                                  "log_file":f"Game-{x}-NNEV3.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"new", 
                                                  "normalize" : False,
                                                  "model_id":2
                                                  }}),
    ]

    gamekwargs = lambda x : {
        "log_file" : f"Game-{x}.log",
        "log_level" : logging.INFO,
        "timeout" : 30,
        "gather_data":True,
        "model_paths":["../Models/ModelMB10-160/model.tflite"]#,"../Models/ModelMB9-125/model.tflite","../Models/ModelMB9-100/model.tflite"]
    }
    
    for i in range(1):
        #print(timeit.timeit("play_games(players, gamekwargs, n=100, cpus=5, chunksize=10,shuffle_player_order=True)",globals=globals(),number=5))
        #act_players = random.sample(players, 4)
        results = play_games(players, gamekwargs, n=100, cpus=10, chunksize=4,shuffle_player_order=True)
        #results = play_games(act_players, gamekwargs, n=1000, cpus=10, chunksize=10,shuffle_player_order=True)
        res = get_loss_percents(results,player="all", show=True)
        print(res)
        N = len([r for r in results if r is not None])
        x = int(res["NNEV"]/100*N)
        p0 = 0.25
        print("N trials = ",N)
        print("X succesfull trial = ",x)
        print("p0 = ",p0)
        # Null hypothesis is that the probability of losing is greater than or equal to 0.25
        # H1 is that the probability of losing is less than 0.25
        p_value = sc_stats.binom.cdf(x,N, p0)
        ci = sc_stats.binom.interval(0.99, N, p0)
        print(p_value)
        print(ci)
        p_value_for_human_game = sc_stats.binom.cdf(7,10, 0.75)
        ci_human = sc_stats.binom.interval(0.95, 10, 0.75)
        print(p_value_for_human_game)
        print(ci_human)
        
        
        