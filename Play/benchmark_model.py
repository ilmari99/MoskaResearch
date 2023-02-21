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
from Simulate import play_games, get_loss_percents
from Utils import make_log_dir
from PlayerWrapper import PlayerWrapper

def player_benchmark1(
        player : PlayerWrapper,
        cpus : int,
        chunksize : int = 4,
        folder : str = "Benchmark1",
        custom_game_kwargs : Dict[str,Any] = {},
                     ) -> float:
    """ Benchmark a player against a set of predefined models.
    """
    models = ["./Models/ModelMB11-260/model.tflite"]
    if "model_paths" in custom_game_kwargs:
        models += custom_game_kwargs["model_paths"]
        custom_game_kwargs.pop("model_paths")
    models = [os.path.abspath(m) for m in models]
    models = list(set(models))
    print("Models: ", models)

    # Game arguments
    gamekwargs = {**{
        "log_file" : "Game-{x}.log",
        "log_level" : logging.DEBUG,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":models,
    }, **custom_game_kwargs}

    # Players shared arguments, except the player or otherwise specified
    shared_kwargs = {
        "log_level" : logging.INFO,
    }
    
    players = [
        player,
        PlayerWrapper(NNEvaluatorBot, {**shared_kwargs,**{"name" : f"ModelMB11-260",
                                    "log_file":"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }}),
        PlayerWrapper(MoskaBot3,{**shared_kwargs,**{"name" : "B3",
                                                  "log_file":"Game-{x}-B3.log"}}),
        PlayerWrapper(HeuristicEvaluatorBot, {**shared_kwargs,**{"name" : f"HEV1",
                                                               "log_file":"Game-{x}-HEV1.log",
                                                               "max_num_states":1000}}),
    ]
    # Make the log directory and change to it
    make_log_dir(folder)
    results = play_games(players, gamekwargs, ngames=1000, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=False)
    loss_perc = get_loss_percents(results)
    print("Benchmark1 done. A great result is < 20% loss")
    os.chdir("..")
    return loss_perc.get(player.settings["name"], 0)


def player_benchmark2(
        player : PlayerWrapper,
        cpus : int,
        chunksize : int = 4,
        folder : str = "Benchmark2",
        custom_game_kwargs : Dict[str,Any] = {},
                     ) -> float:
    """ Benchmark a player against a set of predefined models.
    """
    models = ["./Models/ModelMB11-260/model.tflite"]
    if "model_paths" in custom_game_kwargs:
        models += custom_game_kwargs["model_paths"]
        custom_game_kwargs.pop("model_paths")
    models = [os.path.abspath(m) for m in models]
    models = list(set(models))

    # Game arguments
    gamekwargs = {**{
        "log_file" : "Game-{x}.log",
        "log_level" : logging.DEBUG,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":models,
    }, **custom_game_kwargs}

    # Players shared arguments, except the player or otherwise specified
    shared_kwargs = {
        "log_level" : logging.INFO,
    }
    
    players = [
        player,
        PlayerWrapper(NNEvaluatorBot, {**shared_kwargs,**{"name" : f"ModelMB11-260-1",
                                    "log_file":"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }}),
        PlayerWrapper(NNEvaluatorBot,{**shared_kwargs,**{"name" : f"ModelMB11-260-2",
                                    "log_file":"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }}),
        PlayerWrapper(NNEvaluatorBot, {**shared_kwargs,**{"name" : f"ModelMB11-260-3",
                                    "log_file":"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }}),
    ]
    # Make the log directory and change to it
    make_log_dir(folder)
    results = play_games(players, gamekwargs, ngames=1000, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=False)
    loss_perc = get_loss_percents(results)
    print("Benchmark1 done. A great result is < 20% loss")
    os.chdir("..")
    return loss_perc.get(player.settings["name"], 0)



if __name__ == "__main__":
    # Specify the model paths
    game_kwargs = {
        "model_paths" : [os.path.abspath("./Models/ModelNN1/model.tflite")],
    }
    # Specify the player type
    player_type = WideNNEVHEV
    # Specify the player arguments, '{x}' will be replaced by the game number
    coeffs = {"my_cards":6.154,"len_set_my_cards":2.208,"len_my_cards":1.5723,"kopled":-2.99,"missing_card":52.62}
    player_args = {"name" : "player",
                    "log_file":"Game-{x}-player.log",
                    "log_level":logging.DEBUG,
                    "max_num_states":1000,
                    "max_num_samples":100,
                    "pred_format":"new",
                    "model_id":game_kwargs["model_paths"][0],
                    "coefficients":coeffs,
    }
    # 6.15410198,  2.20813565,  1.57294909, -2.99886373, 52.61803385
    player = PlayerWrapper(player_type, player_args)
    player_benchmark1(player,-1,4,custom_game_kwargs=game_kwargs,folder="Test1")
    player_benchmark2(player,-1,4,custom_game_kwargs=game_kwargs)

