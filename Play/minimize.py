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
    gamekwargs = lambda x : {
        "log_file" : f"Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 30,
        "gather_data":True,
        "model_paths":["../Models/ModelMB11-260/model.tflite","../ModelNN1/model.tflite"],
    }
    player_to_minimize = "WIDE"
    print(f"Simulating with params: {params}")
    players = [
        (WideNNEVHEV, lambda x : {**shared_kwargs,**{"name" : f"WIDE",
                                                    "log_file":f"Game-{x}-WIDE.log",
                                                    "coefficients":params,
                                                    "max_num_states":1000,
                                                    "max_num_samples":100,
                                                     "pred_format":"new",
                                                     "model_id":1,
                                                    }}),

        (MoskaBot2, lambda x : {**shared_kwargs,**{"name" : f"B2-1","log_file":f"Game-{x}-B-3.log"}}),

        (NNEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV",
                                                  "log_file":f"Game-{x}-NNEV.log", 
                                                  "max_num_states":1000,
                                                  "pred_format":"old",
                                                  "model_id":0,
                                                  }}),

        (MoskaBot3,lambda x : {**shared_kwargs,**{"name" : f"B3-2","log_file":f"Game-{x}-B-4.log"}}),
    ]
    results = play_games(players, gamekwargs, ngames=1000, cpus=50, chunksize=4,shuffle_player_order=True, verbose=False)
    out = get_loss_percents(results,player=player_to_minimize, show=False)
    print(f"Player '{player_to_minimize}' lost: {out} %")
    return out

def to_minimize_call(log_dir = "Minimize"):
    make_log_dir(log_dir,append=True)
    x0 = [2.3, 0.77, -2, 0.39, 52]
    res = minimize(to_minimize_func,x0=x0,method="Powell",options={"maxiter":250,"disp":True})
    print(f"Minimization result: {res}")
    return

if __name__ == "__main__":
    to_minimize_call()
