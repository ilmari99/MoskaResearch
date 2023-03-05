#!/usr/bin/env python3
from collections import OrderedDict
import logging
import os
import sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from Moska.Player.NNSampleEvaluatorBot import NNSampleEvaluatorBot
from Moska.Player.WideEvaluatorBot import WideEvaluatorBot
from Moska.Player.WideHIFEvaluatorBot import WideHIFEvaluatorBot
import random
import numpy as np
from scipy.optimize import minimize
from Simulate import play_games, get_loss_percents
from Utils import make_log_dir
from PlayerWrapper import PlayerWrapper


def to_minimize_func(params : List, 
                     players : List[PlayerWrapper],
                     param_kwargs : OrderedDict,
                     player_to_minimize : str,
                     ngames=1000,
                     verbose=False,
                     cpus=-1,
                     custom_gamekwargs={}
                     ):
    """ This function is used to minimize the loss of a player over number of games, by changing the parameters of the player.

    Args:
        params (List): The new parameters to use for the player. Scipy minimize will pass this as a list.
        players (List[PlayerWrapper]): The list of players to use in the game.
        param_kwargs (OrderedDict): The parameters to change for the player.
        player_to_minimize (str): The name of the player to minimize the loss of.
        ngames (int, optional): How many games to play on EACH DIFFERENT SET OF PARAMETERS. Defaults to 1000.
        verbose (bool, optional): Whether to output more or less info. Defaults to False.
        cpus (int, optional): Number of CPUS to use. Defaults to all physical CPUS.
        custom_gamekwargs (dict, optional): Custom dictionary of game arguments. Defaults to {}.

    Raises:
        ValueError: If the player_to_minimize is not found in the players list.

    Returns:
        float: The percent loss of the player over ngames.
    """    
    # Replace the values in param_kwargs with the values in params
    for i,keyval in enumerate(param_kwargs.items()):
        key,val = keyval
        param_kwargs[key] = params[i]

    # NOTE: The keyword is 'coefficients' for Wide model, and parameters for ModelBot3 for example
    for i,pl in enumerate(players):
        if pl.settings["name"] == player_to_minimize:
            pl.settings["coefficients"] = param_kwargs
            break
    # This else part is executed if the for loop is not broken; the player is not found
    else:
        raise ValueError(f"Could not find player '{player_to_minimize}' in players list.")

    gamekwargs = {**{
        "log_file" : "Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 30,
        "gather_data":True,
        "model_paths":["../Models/ModelMB11-260/model.tflite","../ModelNN1/model.tflite"]},
        **custom_gamekwargs
    }
    print(f"Simulating with params: {param_kwargs}")
    results = play_games(players, gamekwargs, ngames=ngames, cpus=cpus, chunksize=4,shuffle_player_order=True, verbose=verbose)
    out = get_loss_percents(results,player=player_to_minimize, show=False)
    print(f"Player '{player_to_minimize}' lost: {out} %")
    return out

def to_minimize_call(players : List[PlayerWrapper],
                     param_kwargs : dict,
                     player_to_minimize,
                     log_dir = "Minimize",
                     ngames=1000,
                     verbose=False,
                     cpus=-1,
                     custom_gamekwargs={}
                     ):
    """ The function to call to minimize the loss of a player over number of games, by changing the parameters of the player.

    Args:
        params (List): The new parameters to use for the player. Scipy minimize will pass this as a list.
        players (List[PlayerWrapper]): The list of players to use in the game.
        param_kwargs (OrderedDict): The parameters to change for the player.
        player_to_minimize (str): The name of the player to minimize the loss of.
        ngames (int, optional): How many games to play on EACH DIFFERENT SET OF PARAMETERS. Defaults to 1000.
        verbose (bool, optional): Whether to output more or less info. Defaults to False.
        cpus (int, optional): Number of CPUS to use. Defaults to all physical CPUS.
        custom_gamekwargs (dict, optional): Custom dictionary of game arguments. Defaults to {}.
    """    
    make_log_dir(log_dir,append=True)
    x0 = np.array(list(param_kwargs.values()))
    args = (players,param_kwargs,player_to_minimize,ngames,verbose,cpus,custom_gamekwargs)
    res = minimize(to_minimize_func,x0=x0,method="Powell",options={"maxiter":250,"disp":True},args=args)
    print(f"Minimization result: {res}")
    return

if __name__ == "__main__":
    shared_kwargs = {
        "log_level" : logging.DEBUG,
    }
    # NOTE: The model_paths should be either absolute, or or prefix with "../" to be relative to the current directory
    custom_gamekwargs = {
        "model_paths":[os.path.abspath(path) for path in ["./Models/ModelMB11-260/model.tflite","./Models/ModelNN1/model.tflite"]],
        "gather_data":False
    }
    # These are the parameters to change for the player. This is also the initial quess
    param_kwargs = OrderedDict({"my_cards" : 2.3,
                            "len_set_my_cards" : 0.77,
                            "len_my_cards" : -2,
                            "kopled":0.39,
                            "missing_card" : 52
                            })
    players = [
        PlayerWrapper(WideEvaluatorBot, {**shared_kwargs,**{"name" : f"WIDE",
                                                    "coefficients":{},
                                                    "max_num_states":1000,
                                                    #"max_num_samples":100,
                                                     "pred_format":"new",
                                                     "model_id":1,
                                                    }},infer_log_file=True),

        PlayerWrapper(MoskaBot2, {**shared_kwargs,**{"name" : f"B2-1"}},infer_log_file=True),

        PlayerWrapper(NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV",
                                                  "max_num_states":1000,
                                                  "pred_format":"old",
                                                  "model_id":0,
                                                  }}, infer_log_file=True),

        PlayerWrapper(MoskaBot3,{**shared_kwargs,**{"name" : f"B3-2"}},infer_log_file=True),
    ]

    to_minimize_call(players, param_kwargs,player_to_minimize="WIDE",log_dir="Minimize",ngames=1000,verbose=False,cpus=50,custom_gamekwargs=custom_gamekwargs)
