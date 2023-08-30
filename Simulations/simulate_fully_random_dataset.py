#!/usr/bin/env python3
import os
import argparse
from MoskaEngine.Play.create_dataset import create_dataset
from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NewRandomPlayer import NewRandomPlayer

def get_players(verbose = 0):
    players = [PlayerWrapper(NewRandomPlayer, {}, infer_log_file=True,number=n) for n in range(1,5)]
    if verbose:
        for p in players:
            print(p)
    return players

def simulate_fully_random_dataset(number_of_rounds = 100,
                                  number_of_games = 100,
                                  folder = "FullyRandomDataset",
                                  cpus = None,
                                  chunksize = None,
                                  verbose = 0,
                                  ):
    cpus = os.cpu_count() if cpus is None else cpus
    chunksize = min(int(number_of_games / cpus*10),1) if chunksize is None else chunksize
    folder = os.path.abspath(folder)
    players = get_players(verbose)
    create_dataset(number_of_rounds, number_of_games, folder, cpus,chunksize,verbose=verbose,players=players, nplayers=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulate a fully random dataset.")
    parser.add_argument("--number_of_rounds", help="Number of rounds to simulate",default=2, type=int)
    parser.add_argument("--number_of_games", help="Number of games to simulate",default=5, type=int)
    parser.add_argument("--folder", help="Folder to save the dataset in",default="FullyRandomDataset", type=str)
    parser.add_argument("--cpus", help="Number of cpus to use",default=None, type=int)
    parser.add_argument("--chunksize", help="Chunksize for multiprocessing",default=None, type=int)
    parser.add_argument("--verbose", help="Verbosity level",default=0, type=int)
    args = parser.parse_args()
    simulate_fully_random_dataset(args.number_of_rounds,
                                    args.number_of_games,
                                    args.folder,
                                    args.cpus,
                                    args.chunksize,
                                    args.verbose,
                                    )
                   

