import os
from MoskaEngine.Play.benchmark import BENCH3, clean_up
from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NNEvaluatorBot import NNEvaluatorBot
from MoskaEngine.Player.NNHIFEvaluatorBot import NNHIFEvaluatorBot

good_pl_args = {"model_id":os.path.abspath(f"/home/ilmari/python/MoskaResearch/ModelTfliteFiles/2202_basic_top_V13_5256.tflite"),
                "max_num_states" : 1000,
                "pred_format" : "bitmap",
                "name" : "test_player",
                }

game_kwargs = {
        "model_paths" : [os.path.abspath(f"/home/ilmari/python/MoskaResearch/ModelTfliteFiles/2202_basic_top_V13_5256.tflite")],
}

print(good_pl_args)
print(game_kwargs)
test_player = PlayerWrapper(NNHIFEvaluatorBot, good_pl_args, infer_log_file=True, number=1)



clean_up()
BENCH3.run(test_player, cpus=15, chunksize=1, ngames=1000, custom_game_kwargs=game_kwargs)