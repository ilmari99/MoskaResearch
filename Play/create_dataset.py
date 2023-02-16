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
from utils import make_log_dir, get_random_players

def create_dataset(nrounds : int,
                   num_games : int,
                   folder : str,
                   cpus : int,
                   chunksize : int = 4,
                   use_HIF : bool = False,
                   verbose : bool=True,
                   players = "random"
                   ):
    """Creates a dataset by playing games between random players.
    """

    make_log_dir(folder)
    gamekwargs = lambda x : {
        "log_file" : f"Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 30,
        "gather_data":True,
        "model_paths":["../Models/ModelMB11-260/model.tflite"]
    }
    print(f"Creating dataset with {nrounds} rounds and {num_games} games per round.")
    print(f"Total games: {nrounds*num_games}.")
    print(f"Using {cpus} cpus and chunksize {chunksize}.")
    print(f"Using HIF: {use_HIF}.")
    print(f"Game kwargs: {gamekwargs(0)}")
    time_taken = 0
    for i in range(nrounds):
        start_time = time.time()
        players = get_random_players(4, use_HIF=use_HIF) if players == "random" else players
        print(f"Round {i+1} players: {players}.")
        results = play_games(players, gamekwargs, n=num_games, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=verbose)
        if verbose:
            get_loss_percents(results)
        end_time = time.time()
        time_taken += (end_time - start_time)
        print(f"Round {i+1} took {end_time - start_time} seconds.")
        print(f"Estimated time remaining: {time_taken/(i+1) * (nrounds - i-1)} minutes.")
    print(f"Finished. Total time taken: {time_taken/60} minutes.")
    return

if __name__ == "__main__":
    create_dataset(nrounds=100,num_games=100,folder="Dataset",cpus=8,use_HIF=False,players="random")