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
from Utils import args_to_gamekwargs, make_log_dir,get_random_players
from PlayerWrapper import PlayerWrapper

def get_human_players() -> List[PlayerWrapper]:
    """Returns a list of PlayerWrappers, including 1 human player.
    """
    shared_kwargs = {"log_level" : logging.DEBUG,
                     "delay":1,
                     "requires_graphic":True}
    players = []
    players.append(PlayerWrapper(HumanPlayer, {**shared_kwargs, **{"name":"Human","log_file":"human-{x}.log"}}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV1",
                                            "log_file":"Game-{x}-NNEV1.log", 
                                            "max_num_states":8000,
                                            "max_num_samples":1000,
                                            "pred_format":"new",
                                            }}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV2",
                                            "log_file":"Game-{x}-NNEV2.log", 
                                            "max_num_states":8000,
                                            "max_num_samples":1000,
                                            "pred_format":"new",
                                            }}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV3",
                                            "log_file":"Game-{x}-NNEV3.log", 
                                            "max_num_states":8000,
                                            "max_num_samples":1000,
                                            "pred_format":"new",
                                            }}))
    return players




def play_as_human(game_id = 0):
    #players = get_human_players()
    cwd = os.getcwd()
    players = get_random_players(4)
    gamekwargs = {
        "log_file" : "Humangame-{x}.log",
        "players" : players,
        "log_level" : logging.DEBUG,
        "timeout" : 1000,
        "model_paths":[os.path.abspath(path) for path in ["./Models/ModelMB11-260/model.tflite","./Models/ModelNN1/model.tflite"]]
    }
    game = args_to_gamekwargs(gamekwargs,players,gameid = game_id,shuffle = True)
    make_log_dir("HumanLogs",append=True)
    game = MoskaGame(**game)
    out = game.start()
    os.chdir(cwd)
    return out

if __name__ == "__main__":
    for i in range(10):
        play_as_human(game_id=i)
