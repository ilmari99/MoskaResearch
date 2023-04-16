#!/usr/bin/env python3
import logging
import math
import os
import shutil
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import sys
from Moska.Game.Game import MoskaGame
from Moska.Player.NewRandomPlayer import NewRandomPlayer
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
from Moska.Player.SVDEvaluatorBot import SVDEvaluatorBot
import random
import numpy as np
from scipy.optimize import minimize
from Simulate import play_games, get_loss_percents
from Utils import make_log_dir
from PlayerWrapper import PlayerWrapper

class Benchmark:
    def __init__(self, main_players : Tuple[PlayerWrapper], folder : str, game_kwargs : Dict[str,Any] = {}, shared_kwargs : Dict[str,Any] = {}):
        self.main_players = main_players
        self.folder = folder
        if not game_kwargs:
            game_kwargs = {
                "log_file" : "Game-{x}.log",
                "log_level" : logging.DEBUG,
                "timeout" : 60,
                "gather_data":False,
                "model_paths":[],
            }
        self.game_kwargs = game_kwargs
        self.shared_kwargs = shared_kwargs

    def calc_CI(self, loss_p, ngames):
        d = 1.96 * math.sqrt(loss_p * (1-loss_p) / ngames)
        return (loss_p - d, loss_p + d)
    
    def run(self, player : PlayerWrapper, cpus : int = -1, chunksize : int = 1, ngames : int = 1000, custom_game_kwargs : Dict[str,Any] = {},custom_shared_pl_kwargs : Dict[str,Any] = {}):
        """ Benchmark a player against a set of predefined models.
        """
        custom_game_kwargs = custom_game_kwargs.copy()
        models = self.game_kwargs.copy().get("model_paths", [])
        if "model_paths" in custom_game_kwargs:
            models += custom_game_kwargs["model_paths"]
            custom_game_kwargs.pop("model_paths")
        models = [os.path.abspath(m) for m in models]
        models = list(set(models))

        # Game arguments
        gamekwargs = {**self.game_kwargs, **custom_game_kwargs}
        players = [player] + self.main_players
        for pl in players:
            pl.settings = {**self.shared_kwargs, **pl.settings, **custom_shared_pl_kwargs}

        # Make the log directory and change to it
        make_log_dir(self.folder)
        results = play_games(players, gamekwargs, ngames=ngames, cpus=cpus, chunksize=chunksize,shuffle_player_order=True,verbose=False)
        loss_perc = get_loss_percents(results)
        print(f"Confidence interval of {player.settings['name']}: {self.calc_CI(loss_perc.get(player.settings['name'], 0)/100, ngames)}")
        os.chdir("..")
        return loss_perc.get(player.settings["name"], 0)

BENCH1 = Benchmark(
    main_players=[
        PlayerWrapper(NNEvaluatorBot, {"name" : f"ModelMB11-260",
                                    "log_file":"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old-algbr",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }),
        PlayerWrapper(MoskaBot3,{"name" : "B3", "log_file":"Game-{x}-B3.log"}),
        PlayerWrapper(HeuristicEvaluatorBot, {"name" : f"HEV1","log_file":"Game-{x}-HEV1.log","max_num_states":1000}),
    ],
    folder="Benchmark1",
    game_kwargs={
        "log_file" : "Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":[os.path.abspath("./Models/ModelMB11-260/model.tflite")],
    },
    shared_kwargs={
        "log_level" : logging.WARNING,
    }
)

BENCH2 = Benchmark(
    main_players=[
        PlayerWrapper(NNEvaluatorBot, {"name" : f"ModelMB11-260-1",
                                    "log_file":"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old-algbr",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }),
        PlayerWrapper(NNEvaluatorBot,{"name" : f"ModelMB11-260-2",
                                    "log_file":"Game-{x}-NNEV2.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old-algbr",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }),
        PlayerWrapper(NNEvaluatorBot, {"name" : f"ModelMB11-260-3",
                                    "log_file":"Game-{x}-NNEV3.log", 
                                    "max_num_states":1000,
                                    "pred_format":"old-algbr",
                                    "model_id":os.path.abspath("./Models/ModelMB11-260/model.tflite"),
                                    }),
    ],
    folder="Benchmark2",
    game_kwargs={
        "log_file" : "Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":[os.path.abspath("./Models/ModelMB11-260/model.tflite")],
    },
    shared_kwargs={
        "log_level" : logging.WARNING,
    }
)

BENCH3 = Benchmark(
    main_players=[
        PlayerWrapper(MoskaBot3,{"name" : "B3-1", "log_file":"Game-{x}-B3-1.log"}),
        PlayerWrapper(MoskaBot3,{"name" : "B3-2", "log_file":"Game-{x}-B3-2.log"}),
        PlayerWrapper(MoskaBot3,{"name" : "B3-3", "log_file":"Game-{x}-B3-3.log"}),
    ],
    folder="Benchmark3",
    game_kwargs={
        "log_file" : "Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":[],
    },
    shared_kwargs={
        "log_level" : logging.WARNING,
    }
)

BENCH4 = Benchmark(
    main_players=[
        PlayerWrapper(NewRandomPlayer,{"name" : "R1", "log_file":"Game-{x}-R1.log"}),
        PlayerWrapper(NewRandomPlayer,{"name" : "R2", "log_file":"Game-{x}-R2.log"}),
        PlayerWrapper(NewRandomPlayer,{"name" : "R3", "log_file":"Game-{x}-R3.log"}),
    ],
    folder="Benchmark4",
    game_kwargs={
        "log_file" : "Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":[],
        "print_format" : "basic",
    },
    shared_kwargs={
        "log_level" : logging.WARNING,
    }
)

BENCH5 = Benchmark(
    main_players=[
        PlayerWrapper(NNHIFEvaluatorBot, {"name" : f"NNEV1",
                                    "log_file":"Game-{x}-NNEV1.log", 
                                    "max_num_states":1000,
                                    "max_num_samples":1000,
                                    "pred_format":"bitmap",
                                    "model_id":os.path.abspath("./Models/Model-nn1-BB/model.tflite"),
                                    #"min_player" : "player"
                                    }),
        PlayerWrapper(NNHIFEvaluatorBot,{"name" : f"NNEV2",
                                    "log_file":"Game-{x}-NNEV2.log", 
                                    "max_num_states":1000,
                                    "max_num_samples":1000,
                                    "pred_format":"bitmap",
                                    "model_id":os.path.abspath("./Models/Model-nn1-BB/model.tflite"),
                                    #"min_player" : "player"
                                    }),
        PlayerWrapper(NNHIFEvaluatorBot, {"name" : f"NNEV3",
                                    "log_file":"Game-{x}-NNEV3.log", 
                                    "max_num_states":1000,
                                    "max_num_samples":1000,
                                    "pred_format":"bitmap",
                                    "model_id":os.path.abspath("./Models/Model-nn1-BB/model.tflite"),
                                    #"min_player" : "player"
                                    }),
    ],
    folder="Benchmark5",
    game_kwargs={
        "log_file" : "Game-{x}.log",
        "log_level" : logging.WARNING,
        "timeout" : 60,
        "gather_data":False,
        "model_paths":[os.path.abspath("./Models/Model-nn1-BB/model.tflite")],
    },
    shared_kwargs={
        "log_level" : logging.WARNING,
    }
)

def run_bench5(player_type, pl_args, game_kwargs, cpus = 10, chunksize = 1, ngames = 10, custom_shared_pl_kwargs = {}):
    game_kwargs = { **{
        "model_paths" : [],
        "gather_data" : False,
        "log_level" : logging.WARNING
    },**game_kwargs
    }
    player = PlayerWrapper(player_type, pl_args)
    BENCH5.run(player,cpus = cpus,chunksize=chunksize,ngames=ngames,custom_game_kwargs=game_kwargs,custom_shared_pl_kwargs=custom_shared_pl_kwargs)

def small_benchmark(player_type, pl_args, game_kwargs):
    game_kwargs = { **{
        "model_paths" : [],
        "gather_data" : False,
        "log_level" : logging.WARNING
    },**game_kwargs
    }
    player = PlayerWrapper(player_type, pl_args)
    BENCH1.run(player,cpus = 10,chunksize=1,ngames=10,custom_game_kwargs=game_kwargs)
    BENCH2.run(player,cpus = 10,chunksize=1,ngames=10,custom_game_kwargs=game_kwargs)
    BENCH3.run(player,cpus = 10,chunksize=1,ngames=10,custom_game_kwargs=game_kwargs)
    BENCH4.run(player,cpus = 10,chunksize=1,ngames=10,custom_game_kwargs=game_kwargs)

def standard_benchmark(player_type, pl_args, game_kwargs, cpus=50,chunksize=3):
    game_kwargs = { **{
        "model_paths" : [os.path.abspath("./Models/Model-nn1-BB/model.tflite")],
        "gather_data" : False,
        "log_level" : logging.WARNING
    },**game_kwargs
    }
    player = PlayerWrapper(player_type, pl_args)
    BENCH1.run(player,cpus = cpus,chunksize=chunksize,ngames=2000,custom_game_kwargs=game_kwargs)
    BENCH2.run(player,cpus = cpus,chunksize=chunksize,ngames=2000,custom_game_kwargs=game_kwargs)
    BENCH3.run(player,cpus = cpus,chunksize=chunksize,ngames=2000,custom_game_kwargs=game_kwargs)
    BENCH4.run(player,cpus = cpus,chunksize=chunksize,ngames=2000,custom_game_kwargs=game_kwargs)

def clean_up():
    shutil.rmtree("./Benchmark1", ignore_errors=True)
    shutil.rmtree("./Benchmark2", ignore_errors=True)
    shutil.rmtree("./Benchmark3", ignore_errors=True)
    shutil.rmtree("./Benchmark4", ignore_errors=True)

if __name__ == "__main__":
    clean_up()
    # Specify the model paths
    game_kwargs = {
        "model_paths" : [os.path.abspath("./Models/Model-nn1-BB/model.tflite")],
        "gather_data" : False,
        "log_level" : logging.DEBUG
    }
    # Specify the player type
    player_type = NNHIFEvaluatorBot
    # Specify the player arguments, '{x}' will be replaced by the game number
    #coeffs = {"my_cards":6.154,"len_set_my_cards":2.208,"len_my_cards":1.5723,"kopled":-2.99,"missing_card":52.62}
    player_args = {"name" : "player",
                    "log_file":"Game-{x}-player.log",
                    "log_level":logging.DEBUG,
                    "max_num_states":1000,
                    "max_num_samples":100,
                    "pred_format":"bitmap",
                    "model_id":game_kwargs["model_paths"][0],
                    #"coefficients":"random",
    }
    #run_bench5(player_type, player_args, game_kwargs, cpus = 10, chunksize = 1, ngames = 10, custom_shared_pl_kwargs = {"log_level" : logging.DEBUG})
    small_benchmark(player_type, player_args, game_kwargs)
    #standard_benchmark(player_type, player_args, game_kwargs, cpus=50,chunksize=3)



