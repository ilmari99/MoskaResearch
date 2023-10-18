import os
from MoskaEngine.Play.benchmark import BENCH3, BENCH4, clean_up
from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NNHIFEvaluatorBot import NNHIFEvaluatorBot


def benchmark_model(model_path):
    good_pl_args = {"model_id":os.path.abspath(model_path),
                    "max_num_states" : 1000,
                    "pred_format" : "bitmap",
                    "name" : "test_player",
                    }

    game_kwargs = {
            "model_paths" : [os.path.abspath(model_path)],
    }

    print(good_pl_args)
    print(game_kwargs)
    test_player = PlayerWrapper(NNHIFEvaluatorBot, good_pl_args, infer_log_file=True, number=1)



    clean_up()
    b3_loss = BENCH3.run(test_player, cpus=15, chunksize=1, ngames=2000, custom_game_kwargs=game_kwargs)
    b4_loss = BENCH4.run(test_player, cpus=15, chunksize=1, ngames=2000, custom_game_kwargs=game_kwargs)
    return b3_loss, b4_loss

def get_models_to_benchmark(folder):
    """Return a list of tflite files in the folder.
    """
    h5_files = []
    for file in os.listdir(folder):
        if file.endswith(".tflite") and "conv" not in file:
            h5_files.append(os.path.join(folder, file))
    return h5_files

if __name__ == "__main__":
    model_scores = {}
    models = get_models_to_benchmark("/home/ilmari/python/MoskaResearch/ModelTfliteFiles")
    models = models[::2]
    print(f"Found {len(models)} models")
    # Remove every second model
    for model in models:
        b3_loss, b4_loss = benchmark_model(model)
        model_scores[model] = (b3_loss, b4_loss)
        print(f"Model {model} loss: {b3_loss}, {b4_loss}")
    
    # Show a ranking of the models
    print("Model ranking:")
    # Sort by BENCH3 loss
    print(sorted(model_scores.items(), key=lambda x: x[1][0]))
    