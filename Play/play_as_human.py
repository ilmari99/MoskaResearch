#!/usr/bin/env python3
import logging
import os
import sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
from Moska.Game.Game import MoskaGame
from Moska.Player.HumanPlayer import HumanPlayer
from typing import Any, Callable, Dict, Iterable, List, Tuple
from Moska.Player.NNHIFEvaluatorBot import NNHIFEvaluatorBot
from Utils import args_to_gamekwargs, make_log_dir,get_random_players, replace_setting_values
from PlayerWrapper import PlayerWrapper

def get_human_players(model_path : str = "model.tflite",
                      pred_format : str = "bitmap",
                      all_against_human : bool = False) -> List[PlayerWrapper]:
    """Returns a list of
    - Three identical PlayerWrapper(NNHIFEvaluatorBot)s, using the model found in model_path, with pred_format.
    - One PlayerWrapper(HumanPlayer) that can be controlled by the user.
    Args:
        model_path (str): Path to the model file.
        pred_format (str): The format of the predictions. Can be "bitmap" or "vector".
    Returns:
        List[PlayerWrapper]: A list of PlayerWrappers.
    """
    shared_kwargs = {"log_level" : logging.DEBUG,
                     "delay":1,
                     "requires_graphic":True}
    players : List[PlayerWrapper] = []
    players.append(PlayerWrapper(HumanPlayer, {**shared_kwargs, **{"name":"Human","log_file":"human-{x}.log"}}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV1",
                                            "log_file":"Game-{x}-NNEV1.log", 
                                            "max_num_states":8000,
                                            "max_num_samples":1000,
                                            "pred_format":pred_format,
                                            "model_id":os.path.abspath(model_path),
                                            }}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV2",
                                            "log_file":"Game-{x}-NNEV2.log", 
                                            "max_num_states":8000,
                                            "max_num_samples":1000,
                                            "pred_format":pred_format,
                                            "model_id":os.path.abspath(model_path),
                                            }}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV3",
                                            "log_file":"Game-{x}-NNEV3.log", 
                                            "max_num_states":8000,
                                            "max_num_samples":1000,
                                            "pred_format":pred_format,
                                            "model_id":os.path.abspath(model_path),
                                            }}))
    if all_against_human:
        for pl in players:
            if pl.settings["name"] != "Human":
                pl.settings["min_player"] = "Human"
    return players

def get_test_human_players(model_path : str = "./model.tflite",
                           pred_format : str = "bitmap",
                           all_against_human : bool = False) -> List[PlayerWrapper]:
    """Returns a list of four identical PlayerWrapper(NNHIFEvaluatorBot)s, using the model found in model_path, with pred_format.
    This is used for testing purposes.
    """
    shared_kwargs = {"log_level" : logging.DEBUG,
                     "delay":0.1,
                     "requires_graphic":True,
                     "max_num_states":8000,
                     "max_num_samples":1000,
                     "model_id":os.path.abspath(model_path),
                     "pred_format":pred_format,
                     }
    players = []
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs, **{"name" : "Fake-human",
                                            "log_file":"Game-{x}-NNEV-test.log", 
                                            }}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV1",
                                            "log_file":"Game-{x}-NNEV1.log", 
                                            }}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV2",
                                            "log_file":"Game-{x}-NNEV2.log", 
                                            }}))
    players.append(PlayerWrapper(NNHIFEvaluatorBot, {**shared_kwargs,**{"name" : "NNEV3",
                                            "log_file":"Game-{x}-NNEV3.log", 
                                            }}))
    if all_against_human:
        for pl in players:
            if pl.settings["name"] != "Fake-human":
                pl.settings["min_player"] = "Fake-human"
    return players



def get_next_game_id(path : str, filename : str) -> int:
    """Returns the next available game id, by checking which files exist in the given path.
    """
    if "{x}" not in filename:
        raise ValueError("Filename must contain '{x}'")
    # if the folder does not exist, return 0
    if not os.path.exists(path):
        return 0
    # Pick any file that exists
    unique_filename = os.listdir(path)[0]
    i = -1
    while os.path.exists(os.path.join(path, unique_filename)):
        i += 1
        unique_filename = replace_setting_values({"filename" : filename},game_id = i)["filename"]
    print("Next game id",i)
    return i


def play_as_human(model_path = "./Models/Model-nn1-fuller/model.tflite",
                  pred_format="bitmap",
                  all_against_human=False,
                  test=False):
    """ Play as a human against three NNHIFEvaluatorBots. The order of the players is shuffled.
    """
    # Get a list of players
    if test:
        players = get_test_human_players(model_path = model_path, pred_format=pred_format, all_against_human=all_against_human)
    else:
        players = get_human_players(model_path = model_path, pred_format=pred_format, all_against_human=all_against_human)
    #players = get_test_human_players(model_path = "./Models/Model-nn1-fuller/model.tflite", pred_format="bitmap")
    cwd = os.getcwd()
    folder = "HumanLogs"
    # Find the next available game id
    game_id = get_next_game_id("./" + folder,"HumanGame-{x}.log")
    gamekwargs = {
        "log_file" : "HumanGame-{x}.log",
        "players" : players,
        "log_level" : logging.INFO,
        "timeout" : 2000,
        "model_paths":[os.path.abspath(path) for path in [model_path]],
        "player_evals" : "plot",
        "print_format" : "human",
        "to_console" : True,
    }
    # Convert general game arguments to game specific arguments (replace '{x}' with game_id)
    game_args = args_to_gamekwargs(gamekwargs,players,gameid = game_id,shuffle = True)
    # Changes to the log directory for the duration of the game
    make_log_dir(folder,append=True)
    game = MoskaGame(**game_args)
    out = game.start()
    os.chdir(cwd)
    return out

if __name__ == "__main__":
    play_as_human(model_path="./Models/Model-nn1-BB/model.tflite", pred_format="bitmap",test=False, all_against_human=True)
