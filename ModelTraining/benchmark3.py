import os
from MoskaEngine.Play.benchmark import BENCH3, clean_up
from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NNEvaluatorBot import NNEvaluatorBot

good_pl_args = {"model_id":os.path.abspath(f"./Models/model-basic-from-1000k-games.tflite"),
                "max_num_states" : 1000,
                "pred_format" : "bitmap",
                }

game_kwargs = {
        "model_paths" : [os.path.abspath(f"./Models/model-basic-from-1000k-games.tflite")],
}

print(good_pl_args)
print(game_kwargs)
test_player = PlayerWrapper(NNEvaluatorBot, good_pl_args, infer_log_file=True, number=1)



clean_up()
BENCH3.run(test_player, cpus=15, chunksize=1, ngames=200, custom_game_kwargs=game_kwargs)