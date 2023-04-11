#!/usr/bin/env python3
import logging
import os
import sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
from Moska.Game.Game import MoskaGame
from Moska.Player.MoskaBot3 import MoskaBot3
from Moska.Player.AbstractPlayer import AbstractPlayer
from Moska.Player.HumanPlayer import HumanPlayer
from typing import Any, Callable, Dict, Iterable, List, Tuple, TYPE_CHECKING
from Moska.Player.MoskaBot2 import MoskaBot2
from Moska.Player.MoskaBot0 import MoskaBot0
from Moska.Player.MoskaBot1 import MoskaBot1
from Moska.Player.RandomPlayer import RandomPlayer
from Moska.Player.NewRandomPlayer import NewRandomPlayer
from Moska.Player.NNEvaluatorBot import NNEvaluatorBot
from Moska.Player.NNHIFEvaluatorBot import NNHIFEvaluatorBot
from Moska.Player.HeuristicEvaluatorBot import HeuristicEvaluatorBot
if TYPE_CHECKING:
    from PlayerWrapper import PlayerWrapper
import random
from scipy.optimize import minimize

"""
This file contains utility functions for playing (multiple) games of Moska.
"""


def replace_setting_values(settings : Dict[str,Any], game_id : int = 0) -> Dict[str,Any]:
    """ Create a new settings dict, with the game id replaced.
    """
    # Create a new dict, so that the original settings are not changed.
    instance_settings = settings.copy()
    # Replace the game id in the new settings if the value is a string.
    for key,val in instance_settings.items():
        if isinstance(val, str):
            instance_settings[key] = instance_settings[key].replace("{x}", str(game_id))
        elif key == "model_paths":
            if isinstance(val, list):
                instance_settings[key] = [os.path.abspath(p) for p in val]
            elif isinstance(val, str):
                instance_settings[key] = os.path.abspath(val)
        elif key == "model_id" and val != "all":
            if isinstance(val, list):
                for i in range(len(val)):
                    if isinstance(val[i], str):
                        instance_settings[key][i] = os.path.abspath(val[i])
            elif isinstance(val, str):
                instance_settings[key] = os.path.abspath(val)
    return instance_settings



def make_log_dir(folder : str, append : bool = False) -> None:
    """Makes a log directory if it doesn't exist.
    Also changes the working directory to the log directory.

    Args:
        log_dir (str): The directory to make
        append (bool): If the directory already exists, should we append to it or raise an exception. Defaults to False.
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    elif not append:
        raise Exception(f"Folder '{folder}' already exists.")
    os.chdir(folder + "/")
    if not os.path.isdir("Vectors"):
        os.mkdir("Vectors")

def args_to_gamekwargs(
    game_kwargs : dict,
    players : List['PlayerWrapper'],
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
    game_args = replace_setting_values(game_kwargs, gameid)
    players = [player(gameid) for player in players]
    if not players:
        assert "nplayers" in game_args or "players" in game_args
    else:
        game_args["players"] = players
    if shuffle and "players" in game_args:
        random.shuffle(game_args["players"])
    return game_args

def get_four_players_that_are_NewRandomPlayers():
    """ Return a list of NewRandomPlayers in PlayerWrappers"""
    # Due to import conflict
    from PlayerWrapper import PlayerWrapper
    shared_kwargs = {
        "log_level" : logging.WARNING,
    }
    players = [
        PlayerWrapper(NewRandomPlayer, {**shared_kwargs,**{"name" : f"NewRandomPlayer-1"}},infer_log_file=True),
        PlayerWrapper(NewRandomPlayer, {**shared_kwargs,**{"name" : f"NewRandomPlayer-2"}},infer_log_file=True),
        PlayerWrapper(NewRandomPlayer, {**shared_kwargs,**{"name" : f"NewRandomPlayer-3"}},infer_log_file=True),
        PlayerWrapper(NewRandomPlayer, {**shared_kwargs,**{"name" : f"NewRandomPlayer-4"}},infer_log_file=True),
    ]
    return players

def get_four_same_players(pl_type, pl_kwargs):
    """ Return a list of PlayerWrappers with the same parameters.
    """
    # Due to import conflict
    from PlayerWrapper import PlayerWrapper
    default_kwargs = {
        "log_level" : logging.DEBUG,
        "delay" : 0,
    }
    pl_name = pl_type.__name__
    players = [
        PlayerWrapper(pl_type, {**default_kwargs,**{"name" : pl_name + "-1",**pl_kwargs}},infer_log_file=True),
        PlayerWrapper(pl_type, {**default_kwargs,**{"name" : pl_name + "-2",**pl_kwargs}},infer_log_file=True),
        PlayerWrapper(pl_type, {**default_kwargs,**{"name" : pl_name + "-3",**pl_kwargs}},infer_log_file=True),
        PlayerWrapper(pl_type, {**default_kwargs,**{"name" : pl_name + "-4",**pl_kwargs}},infer_log_file=True),
    ]
    return players


def get_random_players(nplayers : int, shared_kwargs : Dict = {}, use_HIF : bool = False, infer_log_file = False) -> List['PlayerWrapper']:
    """ Return a list of PlayerWrappers with random parameters.
    """
    # Due to import conflict
    from PlayerWrapper import PlayerWrapper
    shared_kwargs_default = {
        "log_level" : logging.WARNING,
    }
    shared_kwargs = {**shared_kwargs_default, **shared_kwargs}
    
    nn_models = [os.path.abspath(p) for p in ["./Models/ModelNN1/model.tflite","./Models/Model-nn1-BB/model.tflite"]]
    nn_models = [(path, fmt) for path,fmt in zip(nn_models,["new-algbr","bitmap"])]

    # NOTE: The players logs might not be correct for the game index, to reduce the number of files
    much_players = [
        (MoskaBot3, {**shared_kwargs,**{"name" : f"B3-1","parameters":"random"}}),
        (MoskaBot3, {**shared_kwargs,**{"name" : f"B3-2","parameters":"random"}}),
        (MoskaBot3, {**shared_kwargs,**{"name" : f"B3-3","parameters":"random"}}),
        (MoskaBot3, {**shared_kwargs,**{"name" : f"B3-4","parameters":"random"}}),
        (MoskaBot3, {**shared_kwargs,**{"name" : f"B3-5","parameters":"random"}}),
        
        (MoskaBot2, {**shared_kwargs,**{"name" : f"B2-1","parameters":"random"}}),
        (MoskaBot2, {**shared_kwargs,**{"name" : f"B2-2","parameters":"random"}}),
        (MoskaBot2, {**shared_kwargs,**{"name" : f"B2-3","parameters":"random"}}),
        (MoskaBot2, {**shared_kwargs,**{"name" : f"B2-4","parameters":"random"}}),
        
        (NewRandomPlayer, {**shared_kwargs,**{"name" : f"R1"}}),
        #(NewRandomPlayer, {**shared_kwargs,**{"name" : f"R2"}}),
        
        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV1-old",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[0][1],
                                            "model_id":nn_models[0][0],
                                            }}),
        
        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV2-old",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[0][1],
                                            "model_id":nn_models[0][0],
                                            }}),

        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV3-old",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[0][1],
                                            "model_id":nn_models[0][0],
                                            }}),
        
        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV4-old",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[0][1],
                                            "model_id":nn_models[0][0],
                                            }}),
        
        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV1-new",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[1][1],
                                            "model_id":nn_models[1][0],
                                            }}),
        
        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV2-new",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[1][1],
                                            "model_id":nn_models[1][0],
                                            }}),

        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV3-new",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[1][1],
                                            "model_id":nn_models[1][0],
                                            }}),
        
        (NNEvaluatorBot, {**shared_kwargs,**{"name" : f"NNEV4-new",
                                            "max_num_states":random.randint(1,5000),
                                            "pred_format":nn_models[1][1],
                                            "model_id":nn_models[1][0],
                                            }}),
    ]
    
    if use_HIF:
        much_players.append((NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : f"NNHIFEV",
                                            "max_num_states":random.randint(1,10000),
                                            "max_num_samples":random.randint(10,1000),
                                            "pred_format":"new-algbr",
                                            "model_id":nn_models[1],
                                            }}))
        
        much_players.append((NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : f"NNHIFEV2",
                                            "max_num_states":random.randint(1,10000),
                                            "max_num_samples":random.randint(10,1000),
                                            "pred_format":"new-algbr",
                                            "model_id":nn_models[1],
                                            }}))
        much_players.append((NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : f"NNHIFEV3",
                                            "max_num_states":random.randint(1,10000),
                                            "max_num_samples":random.randint(10,1000),
                                            "pred_format":"new-algbr",
                                            "model_id":nn_models[1],
                                            }}))
    
    acting_players = random.sample(much_players, nplayers)
    acting_players = [PlayerWrapper(player, kwargs,infer_log_file=infer_log_file) for player, kwargs in acting_players]
    if not infer_log_file:
        for player in acting_players:
            player.settings["log_file"] = player.settings.get("name",str(player.player_class.__name__)) + ".log"
    return acting_players
