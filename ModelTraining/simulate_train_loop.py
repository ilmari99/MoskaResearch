import os
import random
import sys
import tensorflow as tf
from read_to_dataset import read_to_dataset
from MoskaEngine.Play.create_dataset import create_dataset
from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NewRandomPlayer import NewRandomPlayer
from MoskaEngine.Player.NNEvaluatorBot import NNEvaluatorBot

""" Continuosly simulate games (starting from random), improve/create model, simulate games, improve model, ... etc.
"""
TRAINING_EPOCH_NUMBER = 530
SIMULATE_EPOCH_NUMBER = 106
BATCH_SIZE = 4096*2

def get_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(600, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    
    x = tf.keras.layers.Dense(550, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    
    x = tf.keras.layers.Dense(450, activation="relu")(x)
    x = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    return model

def get_players(only_random = False):
    n_random = 4 if only_random else 0
    n_good_players = 4 - n_random
    random_players = [PlayerWrapper(NewRandomPlayer, {}, infer_log_file=True,number=n) for n in range(1,n_random+1)]
    good_pl_args = {"model_id":os.path.abspath(f"model_{SIMULATE_EPOCH_NUMBER}.tflite") if not only_random else None,
                    "max_num_states" : 100,
                    "pred_format" : "bitmap",
                    "noise_level" : 0.0001
                    }
    good_players = [PlayerWrapper(NNEvaluatorBot, good_pl_args, infer_log_file=True,number=n) for n in range(1,n_good_players+1)]
    players = random_players + good_players
    return players

def simulate_games(only_random = False):
    global SIMULATE_EPOCH_NUMBER
    players = get_players(only_random=only_random)
    game_kwargs = {"log_file" : "Game-{x}.log",
                "log_level" : 0,
                "timeout" : 10,
                "gather_data":True,
                "model_paths":[os.path.abspath(f"model_{SIMULATE_EPOCH_NUMBER}.tflite")] if not only_random else [],
                }
    SIMULATE_EPOCH_NUMBER += 1
    create_dataset(1,700, folder=f"FullyRandomDataset_{SIMULATE_EPOCH_NUMBER}", cpus=15, chunksize=4, verbose=0, players=players,gamekwargs=game_kwargs)
    return
    
def train_model(model):
    global TRAINING_EPOCH_NUMBER
    ds = read_to_dataset([f"./FullyRandomDataset_{SIMULATE_EPOCH_NUMBER}/Vectors"],add_channel=False,shuffle_files=True)
    val_ds = ds.take(1000).batch(1000)
    ds = ds.skip(1000).batch(BATCH_SIZE)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_log",histogram_freq=2)
    history = model.fit(ds,epochs=TRAINING_EPOCH_NUMBER + 5, initial_epoch=TRAINING_EPOCH_NUMBER, validation_data=val_ds, callbacks=[tensorboard_cb])
    model.save(f"model_{SIMULATE_EPOCH_NUMBER}.h5")
    success = os.system(f"./Utilities/convert-to-tflite.py model_{SIMULATE_EPOCH_NUMBER}.h5 --output_file model_{SIMULATE_EPOCH_NUMBER}.tflite")
    if success != 0:
        print("Error converting to tflite. Exiting.")
        sys.exit(1)
    os.system(f"rm -r FullyRandomDataset_{SIMULATE_EPOCH_NUMBER}")
    TRAINING_EPOCH_NUMBER += 5
    return history


if __name__ == "__main__":
    #model = get_model((442,))
    model = tf.keras.models.load_model("model_100.h5")
    histories = []
    histories_file = "histories.txt"
    while SIMULATE_EPOCH_NUMBER < 200:
        simulate_games(only_random=True if SIMULATE_EPOCH_NUMBER == 0 else False)
        history = train_model(model)
        histories.append(history)
        with open(histories_file,"a") as f:
            f.write(str(history.history))
            f.write("\n")