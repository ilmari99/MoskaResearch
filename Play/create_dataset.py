#!/usr/bin/env python3
import logging
import os
import sys
import warnings
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
import random
import numpy as np
from scipy.optimize import minimize
from Simulate import play_games, get_loss_percents
from Utils import make_log_dir, get_random_players, get_four_players_that_are_NewRandomPlayers

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
    if os.path.exists(folder):
        warnings.warn(f"Folder {folder} already exists. Overwriting.")
    CWD = os.getcwd()
    model_paths = [os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                      os.path.abspath("./Models/ModelNN1/model.tflite")]
    gamekwargs = {
        "log_file" : "Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 30,
        "gather_data":True,
        #"model_paths":model_paths,
    }
    print(f"Creating dataset with {nrounds} rounds and {num_games} games per round.")
    print(f"Total games: {nrounds*num_games}.")
    print(f"Using {cpus} cpus and chunksize {chunksize}.")
    print(f"Using HIF: {use_HIF}.")
    print(f"Game kwargs: ")
    for k,v in gamekwargs.items():
        print(f"\t{k}: {v}")
    time_taken = 0
    for i in range(nrounds):
        start_time = time.time()
        acting_players = get_random_players(4, use_HIF=use_HIF) if players == "random" else players
        print(f"Round {i+1} players:")
        for p in acting_players:
            print(p)
        make_log_dir(folder,append=True)
        results = play_games(acting_players, gamekwargs, ngames=num_games, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=verbose)
        os.chdir(CWD)
        if not verbose:
            get_loss_percents(results)
        end_time = time.time()
        time_taken += (end_time - start_time)
        print(f"Round {i+1} took {end_time - start_time} seconds.")
        print(f"Estimated time remaining: {time_taken/(i+1) * (nrounds - i-1)} minutes.")
    print(f"Finished. Total time taken: {time_taken/60} minutes.")
    return

if __name__ == "__main__":
    folder = "./Dataset-rand"
    players = get_four_players_that_are_NewRandomPlayers()
    create_dataset(nrounds=10,num_games=10000,folder=folder,cpus=50,use_HIF=False,players=players,verbose=True)
