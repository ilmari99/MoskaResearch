import random
import os
import sys
import argparse

from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NewRandomPlayer import NewRandomPlayer
from MoskaEngine.Player.NNEvaluatorBot import NNEvaluatorBot
from MoskaEngine.Play.create_dataset import create_dataset


def get_players(model_paths, nrandom=1):
    """ Create a list of players from which 4 are chosen each simulation round
    """
    random_players = [PlayerWrapper(NewRandomPlayer, {}, infer_log_file=True, number=i) for i in range(nrandom)]
    # Create two versions of each model, one with noise and one without
    nn_players = []
    for i, model_path in enumerate(model_paths):
        nn_players.append(PlayerWrapper(NNEvaluatorBot, {"model_id":os.path.abspath(model_path),
                                                             "max_num_states" : 1000,
                                                             "pred_format" : "bitmap",
                                                             "noise_level" : 0,
                                                             }, infer_log_file=True, number=i))
        nn_players.append(PlayerWrapper(NNEvaluatorBot, {"model_id":os.path.abspath(model_path),
                                                                "max_num_states" : 1000,
                                                                "pred_format" : "bitmap",
                                                                "noise_level" : 0.1,
                                                                }, infer_log_file=True, number=i+len(model_paths)))
    
    players = random_players + nn_players
    print("Players: ",players)
    return players
    

def simulate_games(model_paths, nrounds=1, ngames=5000, folder="NotRandomDataset_1", nrandom=1, cpus=15, chunksize=3, verbose=1):
    players = get_players(model_paths, nrandom=nrandom)
    game_kwargs = {"log_file" : "Game-{x}.log",
                "log_level" : 0,
                "timeout" : 40,
                "gather_data":True,
                "model_paths":[os.path.abspath(model_path) for model_path in model_paths],
                }
    # Remove all log files from previous simulations
    os.system(f"rm {folder}/*.log")
    create_dataset(nrounds,ngames, folder=folder, cpus=cpus, chunksize=chunksize, verbose=verbose, players=players, gamekwargs=game_kwargs)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulate games")
    parser.add_argument("--nrounds", help="The number of rounds to simulate", default=1)
    parser.add_argument("--ngames", help="The number of games to simulate", default=5000)
    parser.add_argument("--folder", help="The folder to store the data in", default="Dataset")
    parser.add_argument("--nrandom", help="The number of random players to use", default=1)
    parser.add_argument("--cpus", help="The number of cpus to use", default=15)
    parser.add_argument("--chunksize", help="The chunksize to use", default=3)
    parser = parser.parse_args()
    MODEL_PATHS = ["./Models/model-basic-from-1000k-games.tflite",
                   "./Models/model-basic-from-700k-games.tflite",
                   #"./Models/model-conv-from-500k-random-games.tflite",
                   "./Models/model-basic-from-1300k-games.tflite",
                   "./Models/model-conv-from-1000k-nrnd-games.tflite",
                   "./Models/model-basic-from-1400k-nrnd-games.tflite",
                   ]
    NROUNDS = int(parser.nrounds)
    NGAMES = int(parser.ngames)
    FOLDER = parser.folder
    print("Model paths: ",MODEL_PATHS)
    print("Number of rounds: ",NROUNDS)
    print("Number of games: ",NGAMES)
    print("Folder: ",FOLDER)
    simulate_games(MODEL_PATHS, nrounds=NROUNDS, ngames=NGAMES, folder=FOLDER, nrandom=int(parser.nrandom), cpus=int(parser.cpus), chunksize=int(parser.chunksize))