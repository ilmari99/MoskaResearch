#!/usr/bin/env python3
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
from Moska.Player.WideNNEVHEV import WideNNEVHEV
import random
import numpy as np
from scipy.optimize import minimize
from scipy import stats as sc_stats

def make_log_dir(folder : str, append : bool = False) -> None:
    """Makes a log directory if it doesn't exist.
    Also changes the working directory to the log directory.

    Args:
        log_dir (str): The directory to make
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    elif not append:
        raise Exception(f"Folder '{folder}' already exists.")
    os.chdir(folder + "/")
    if not os.path.isdir("Vectors"):
        os.mkdir("Vectors")

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


def get_random_players(nplayers, shared_kwargs = {}, use_HIF = False):
    """Returns a list of random players with random settings."""

    shared_kwargs_default = {
        "log_level" : logging.WARNING,
    }
    shared_kwargs = {**shared_kwargs_default, **shared_kwargs}
    
    # NOTE: The players logs might not be correct for the game index, to reduce the number of files
    much_players = [
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV1",
                                                               "log_file":f"Game-{x % 10}-HEV1.log", 
                                                               "max_num_states":random.randint(1,10000),
                                                               "coefficients" : "random"
                                                               }}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV2",
                                                               "log_file":f"Game-{x % 10}-HEV2.log", 
                                                               "max_num_states":random.randint(1,10000),
                                                               "coefficients" : "random"
                                                               }}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV3",
                                                               "log_file":f"Game-{x % 10}-HEV3.log", 
                                                               "max_num_states":random.randint(1,10000),
                                                               "coefficients" : "random"
                                                               }}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV4",
                                                               "log_file":f"Game-{x % 10}-HEV4.log", 
                                                               "max_num_states":random.randint(1,10000),
                                                               "coefficients" : "random"
                                                               }}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-1","log_file":f"Game-{x % 10}-B3-1.log","parameters":"random"}}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-2","log_file":f"Game-{x % 10}-B3-2.log","parameters":"random"}}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-3","log_file":f"Game-{x % 10}-B3-3.log","parameters":"random"}}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-4","log_file":f"Game-{x % 10}-B3-4.log","parameters":"random"}}),
        
        (MoskaBot2,lambda x : {**shared_kwargs,**{"name" : f"B2-1","log_file":f"Game-{x % 10}-B2-1.log","parameters":"random"}}),
        (MoskaBot2,lambda x : {**shared_kwargs,**{"name" : f"B2-2","log_file":f"Game-{x % 10}-B2-2.log","parameters":"random"}}),
        (MoskaBot2,lambda x : {**shared_kwargs,**{"name" : f"B2-3","log_file":f"Game-{x % 10}-B2-3.log","parameters":"random"}}),
        (MoskaBot2,lambda x : {**shared_kwargs,**{"name" : f"B2-4","log_file":f"Game-{x % 10}-B2-4.log","parameters":"random"}}),
        
        (RandomPlayer, lambda x : {**shared_kwargs,**{"name" : f"R1","log_file":f"Game-{x % 10}-R1.log"}}),
        (RandomPlayer, lambda x : {**shared_kwargs,**{"name" : f"R2","log_file":f"Game-{x % 10}-R2.log"}}),
        
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV1",
                                            "log_file":f"Game-{x % 10}-NNEV1.log", 
                                            "max_num_states":random.randint(1,10000),
                                            "pred_format":"old",
                                            "model_id":"all",
                                            }}),
        
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV2",
                                            "log_file":f"Game-{x % 10}-NNEV2.log", 
                                            "max_num_states":random.randint(1,10000),
                                            "pred_format":"old",
                                            "model_id":"all",
                                            }}),
        
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV3",
                                                    "log_file":f"Game-{x % 10}-NNEV3.log", 
                                                    "max_num_states":random.randint(1,10000),
                                                    "pred_format":"old",
                                                    "model_id":"all",
                                                    }}),
        
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV4",
                                            "log_file":f"Game-{x % 10}-NNEV4.log", 
                                            "max_num_states":random.randint(1,10000),
                                            "pred_format":"old",
                                            "model_id":"all",
                                            }}),
    ]
    
    if use_HIF:
        much_players.append((NNHIFEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNHIFEV",
                                            "log_file":f"Game-{x % 10}-NNHIFEV.log",
                                            "max_num_states":random.randint(1,10000),
                                            "max_num_samples":random.randint(10,1000),
                                            "pred_format":"old",
                                            "model_id":"all",
                                            }}))
        
        much_players.append((NNHIFEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNHIFEV2",
                                            "log_file":f"Game-{x % 10}-NNHIFEV2.log",
                                            "max_num_states":random.randint(1,10000),
                                            "max_num_samples":random.randint(10,1000),
                                            "pred_format":"old",
                                            "model_id":"all",
                                            }}))
        much_players.append((NNHIFEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNHIFEV3",
                                            "log_file":f"Game-{x % 10}-NNHIFEV3.log",
                                            "max_num_states":random.randint(1,10000),
                                            "max_num_samples":random.randint(10,1000),
                                            "pred_format":"old",
                                            "model_id":"all",
                                            }}))
    
    players = random.sample(much_players, nplayers)
    return players