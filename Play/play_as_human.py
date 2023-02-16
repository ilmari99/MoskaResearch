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
from utils import args_to_gamekwargs, make_log_dir




def play_as_human(game_id = 0):
    shared_kwargs = {
        "log_level" : logging.DEBUG,
        "delay":1,
    }
    players = [
        (HumanPlayer,lambda x : {"name":"Human","log_file":f"human-{x}.log","requires_graphic":True}),
        (NNHIFEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV1","requires_graphic":True,
                                                  "log_file":f"Game-{x}-NNEV1.log", 
                                                  "max_num_states":8000,
                                                  "max_num_samples":1000,
                                                  "pred_format":"new",
                                                  }}),
        (NNHIFEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV2","requires_graphic":True,
                                                  "log_file":f"Game-{x}-NNEV2.log", 
                                                  "max_num_states":8000,
                                                  "max_num_samples":1000,
                                                  "pred_format":"new",
                                                  }}),
        (NNHIFEvaluatorBot, lambda x : {**shared_kwargs,**{"name" : f"NNEV3","requires_graphic":True,
                                                  "log_file":f"Game-{x}-NNEV3.log", 
                                                  "max_num_states":8000,
                                                  "max_num_samples":1000,
                                                  "pred_format":"new",
                                                  }}),
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

if __name__ == "__main__":
    make_log_dir("HumanLogs")
    play_as_human()