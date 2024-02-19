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
    
    get_player = lambda model_path, number: PlayerWrapper(NNEvaluatorBot, {"model_id":os.path.abspath(model_path),
                                                             "max_num_states" : 1000,
                                                             "pred_format" : "bitmap",
                                                             "top_p_play" : 1,
                                                             "top_p_weights" : "weighted",
                                                             }, infer_log_file=True, number=number)
    
    total_num_players = sum([model_paths[model_path] for model_path in model_paths]) + nrandom
    for i, model_path in enumerate(model_paths):
        num_players = model_paths[model_path]
        # Can't have same number
        nn_players += [get_player(model_path, j + len(nn_players)) for j in range(num_players)]
        
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
    MODEL_PATHS = {"./ModelTfliteFiles/0311_basic_top_V5V7-1V10V11_5109.tflite" : 3,
                   "./ModelTfliteFiles/2810_basic_notop_V8V9V10V11_4825.tflite" : 3,
                   "./ModelTfliteFiles/0910_basic_top_V6V7V7-1_5668.tflite" : 3,
    }
    NROUNDS = int(parser.nrounds)
    NGAMES = int(parser.ngames)
    FOLDER = parser.folder
    print("Model paths: ",MODEL_PATHS)
    print("Number of rounds: ",NROUNDS)
    print("Number of games: ",NGAMES)
    print("Folder: ",FOLDER)
    simulate_games(MODEL_PATHS, nrounds=NROUNDS, ngames=NGAMES, folder=FOLDER, nrandom=int(parser.nrandom), cpus=int(parser.cpus), chunksize=int(parser.chunksize))