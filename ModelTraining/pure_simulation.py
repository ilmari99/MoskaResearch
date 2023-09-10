import random
import os
import sys
import argparse

from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NewRandomPlayer import NewRandomPlayer
from MoskaEngine.Player.NNEvaluatorBot import NNEvaluatorBot
from MoskaEngine.Play.create_dataset import create_dataset


def get_players(model_path, only_random = False):
    n_random = 4 if only_random else random.randint(0,2)
    n_good_players = 4 - n_random
    random_players = [PlayerWrapper(NewRandomPlayer, {}, infer_log_file=True,number=n) for n in range(1,n_random+1)]
    good_pl_args = {"model_id":os.path.abspath(model_path) if not only_random else None,
                    "max_num_states" : 500,
                    "pred_format" : "bitmap",
                    "noise_level" : 0.01,
                    }
    good_players = [PlayerWrapper(NNEvaluatorBot, good_pl_args, infer_log_file=True,number=n) for n in range(1,n_good_players+1)]
    players = random_players + good_players
    return players

def simulate_games(model_path, only_random = False, nrounds=1, ngames=5000, folder="NotRandomDataset_1"):
    players = get_players(model_path, only_random=only_random)
    game_kwargs = {"log_file" : "Game-{x}.log",
                "log_level" : 0,
                "timeout" : 40,
                "gather_data":True,
                "model_paths":[os.path.abspath(model_path)] if not only_random else [],
                }
    # Remove all log files from previous simulations
    os.system(f"rm {folder}/*.log")
    create_dataset(nrounds,ngames, folder=folder, cpus=15, chunksize=3, verbose=0, players=players,gamekwargs=game_kwargs)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulate games")
    parser.add_argument("model_path", help="The path to the model to simulate")
    parser.add_argument("--only_random", help="Only simulate against random players", action="store_true")
    parser.add_argument("--nrounds", help="The number of rounds to simulate", default=1)
    parser.add_argument("--ngames", help="The number of games to simulate", default=5000)
    parser.add_argument("--folder", help="The folder to store the data in", default="Dataset")
    parser = parser.parse_args()
    MODEL_PATH = parser.model_path
    ONLY_RANDOM = parser.only_random
    NROUNDS = int(parser.nrounds)
    NGAMES = int(parser.ngames)
    FOLDER = parser.folder
    simulate_games(MODEL_PATH, ONLY_RANDOM, NROUNDS, NGAMES, FOLDER)