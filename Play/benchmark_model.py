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

    models = ["./Models/ModelMB11-260/model.tflite", "./Models/ModelNN1/model.tflite"] + models
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
        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV1",
                                    "log_file":f"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old",
                                    "model_id":0,
                                    }}),
        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3","log_file":f"Game-{x}-B3.log"}}),
        (HeuristicEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"HEV1","log_file":f"Game-{x}-HEV1.log", "max_num_states":1000}}),
    ]
    make_log_dir(folder, append=append)
    results = play_games(players, gamekwargs, ngames=num_games, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=False)
    get_loss_percents(results)
    os.chdir("..")


def player_benchmark_random(
        player : Tuple[AbstractPlayer, Callable[[int], Dict[str, Any]]],
        models : List[str],
        num_games : int,
        cpus : int,
        folder : str = "BenchmarkRandom",
        chunksize : int = 4,
        append : bool =False
                     ) -> None:
    
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
        (RandomPlayer,lambda x : {**shared_kwargs,**{"name" : f"Random1","log_file":f"Game-{x}-Random1.log"}}),
        (RandomPlayer,lambda x : {**shared_kwargs,**{"name" : f"Random2","log_file":f"Game-{x}-Random2.log"}}),
        (RandomPlayer,lambda x : {**shared_kwargs,**{"name" : f"Random3","log_file":f"Game-{x}-Random3.log"}}),
    ]

    make_log_dir(folder, append=append)
    results = play_games(players, gamekwargs, ngames=num_games, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=False)
    get_loss_percents(results)
    os.chdir("..")



if __name__ == "__main__":

    model_path = os.path.abspath("./Models/ModelNN1/model.tflite")
    player = (NNEvaluatorBot, lambda x : {"name" : f"ModelNN1",
                                    "log_file":f"Game-{x}-ModelNN1.log",
                                    "max_num_states":1000,
                                    "pred_format":"new",
                                    "model_id":model_path,
    })
    models = [model_path]


    player_benchmark1(player, models, 100, 5, folder="Compare", append=False)
    player_benchmark_random(player, models, 100, 5, folder="Compare", append=True)