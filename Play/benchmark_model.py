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
from simulate import play_games, get_loss_percents
from utils import make_log_dir

def player_benchmark1(
        player : Tuple[AbstractPlayer, Callable[[int], Dict[str, Any]]],
        models : List[str],
        num_games : int,
        cpus : int,
        folder : str = "Benchmark1",
        chunksize : int = 4,
        append : bool =False
                     ) -> None:
    """ Benchmark a player against a set of predefined models.
    """

    # The performance is compared against one MB11-260 model.
    models = ["./Models/ModelMB11-260/model.tflite"] + models
    models = [os.path.abspath(m) for m in models]
    models = list(set(models))

    # Game arguments
    gamekwargs = lambda x : {
        "log_file" : f"Game-{x}.log",
        "log_level" : logging.DEBUG,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":models
    }

    # Players shared arguments, except the player or otherwise specified
    shared_kwargs = {
        "log_level" : logging.INFO,
    }
    
    players = [
        player,
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"ModelMB11-260",
                                    "log_file":f"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("../Models/ModelMB11-260/model.tflite"),
                                    }}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3",
                                                  "log_file":f"Game-{x}-B3.log"}}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV1",
                                                               "log_file":f"Game-{x}-HEV1.log",
                                                               "max_num_states":1000}}),
    ]
    # Make the log directory and change to it
    make_log_dir(folder, append=append)
    results = play_games(players, gamekwargs, ngames=num_games, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=False)
    get_loss_percents(results)
    print("Benchmark1 done. A great result is < 20% loss")
    os.chdir("..")


def player_benchmark2(
        player : Tuple[AbstractPlayer, Callable[[int], Dict[str, Any]]],
        models : List[str],
        num_games : int,
        cpus : int,
        folder : str = "Benchmark2",
        chunksize : int = 4,
        append : bool = False
                     ) -> None:
    
    # The performance is compared against 3 x MB11-260 model.
    models = ["./Models/ModelMB11-260/model.tflite"] + models
    models = [os.path.abspath(m) for m in models]
    models = list(set(models))

    gamekwargs = lambda x : {
        "log_file" : f"Game-{x}.log",
        "log_level" : logging.DEBUG,
        "timeout" : 30,
        "gather_data":True,
        "model_paths":models
    }
    shared_kwargs = {
        "log_level" : logging.INFO,
    }
    
    players = [
        player,
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"ModelMB11-260-1",
                                    "log_file":f"Game-{x}-MB11-260-1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("../Models/ModelMB11-260/model.tflite"),
                                    }}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"ModelMB11-260-2",
                                    "log_file":f"Game-{x}-MB11-260-2.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("../Models/ModelMB11-260/model.tflite"),
                                    }}),
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"ModelMB11-260-3",
                                    "log_file":f"Game-{x}-MB11-260-3.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("../Models/ModelMB11-260/model.tflite"),
                                    }}),
    ]

    make_log_dir(folder, append=append)
    results = play_games(players, gamekwargs, ngames=num_games, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=False)
    get_loss_percents(results)
    print("Benchmark2 done. A great result is < 24% loss")
    os.chdir("..")



if __name__ == "__main__":
    # Specify the model paths
    models = [
        os.path.abspath("./Models/ModelNN1/model.tflite"),
    ]
    # Specify the player type
    player_type = WideNNEVHEV
    # Specify the player arguments, '{x}' will be replaced by the game number
    player_args = {"name" : "player",
                    "log_file":"Game-{x}-player.log",
                    "log_level":logging.DEBUG,
                    "max_num_states":1000,
                    "max_num_samples":100,
                    "pred_format":"new",
                    "model_id":models[0],
    }

    player = (player_type, lambda x : {arg:val if not isinstance(val,str) else val.replace("{x}",str(x)) for arg,val in player_args.items()})
    player_benchmark1(player, models, 10, 5, chunksize=1, folder="Compare", append=False)
    player_benchmark2(player, models, 10, 5, chunksize=1, folder="Compare", append=True)

