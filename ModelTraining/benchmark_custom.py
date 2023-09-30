import os
from MoskaEngine.Play.benchmark import clean_up, Benchmark
from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NNEvaluatorBot import NNEvaluatorBot

previous_player_args = {
        "model_id":os.path.abspath(f"./Models/model-basic-from-1000k-games.tflite"),
        "max_num_states" : 1000,
        "pred_format" : "bitmap",
        }

previous_players = [PlayerWrapper(NNEvaluatorBot, previous_player_args, infer_log_file=True, number=i) for i in range(3)]

test_player_args = {
        "model_id":os.path.abspath(f"./Models/model-basic-from-1300k-games.tflite"),
        "max_num_states" : 1000,
        "pred_format" : "bitmap",
        "name" : "test_player",
        "log_file" : "test_player.log",
        }

test_player = PlayerWrapper(NNEvaluatorBot, test_player_args, infer_log_file=False)

game_kwargs = {
        "model_paths" : [os.path.abspath(f"./Models/model-basic-from-1000k-games.tflite"),
                         os.path.abspath(f"./Models/model-basic-from-1300k-games.tflite")],
        "log_file" : "Game-{x}.log",
        "log_level" : 0,
        "timeout" : 40,
        "gather_data":False,
}

benchmark = Benchmark(previous_players, "custom_benchmark", game_kwargs)

benchmark.run(test_player, cpus=15, chunksize=1, ngames=200, custom_game_kwargs=game_kwargs)