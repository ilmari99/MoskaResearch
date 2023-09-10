import os
import random
import re
import sys
import tensorflow as tf
from read_to_dataset import read_to_dataset
from MoskaEngine.Play.create_dataset import create_dataset
from MoskaEngine.Play.PlayerWrapper import PlayerWrapper
from MoskaEngine.Player.NewRandomPlayer import NewRandomPlayer
from MoskaEngine.Player.NNEvaluatorBot import NNEvaluatorBot

""" Continuosly simulate games (starting from random), improve/create model, simulate games, improve model, ... etc.
"""
SIMULATING_GAMES_FLAG = False

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

def get_branched_model(input_shape):
    """
    This model is a branched neural network, where we use a convolutional network to extract features from the cards,
    and a dense network to extract features from the game info.
    Finally we concatenate the outputs of the two networks and feed them into a dense network.
    """
    info_indices = list(range(0,26))
    card_indices = list(range(26,442))
    inputs = tf.keras.Input(shape=input_shape)
    info = tf.keras.layers.Lambda(lambda x: tf.gather(x, info_indices, axis=1))(inputs)
    cards = tf.keras.layers.Lambda(lambda x: tf.gather(x, card_indices, axis=1))(inputs)
    
    # Information network (26)
    infox = tf.keras.layers.BatchNormalization()(info)
    
    # Card network (must add a channel dimension) (416 elems)
    cards = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(cards)
    cardsx = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, activation="relu")(cards)
    cardsx = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation="relu")(cardsx)
    cardsx = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=1, activation="relu")(cardsx)
    cardsx = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation="relu")(cardsx)
    cardsx = tf.keras.layers.Flatten()(cardsx)
    cardsx = tf.keras.layers.Dense(75, activation="relu")(cardsx)
    cardsx = tf.keras.layers.Dropout(rate=0.4)(cardsx)
    cardsx = tf.keras.layers.Dense(75, activation="relu")(cardsx)
    
    combinedx = tf.keras.layers.concatenate([infox, cardsx])
    combinedx = tf.keras.layers.Dense(120, activation="relu")(combinedx)
    combinedx = tf.keras.layers.Dropout(rate=0.4)(combinedx)
    combinedx = tf.keras.layers.Dense(100, activation="relu")(combinedx)
    combinedx = tf.keras.layers.Dropout(rate=0.4)(combinedx)
    combinedx = tf.keras.layers.Dense(100, activation="relu")(combinedx)
    combinedx = tf.keras.layers.Dense(units=1, activation="sigmoid")(combinedx)
    
    model = tf.keras.Model(inputs=inputs, outputs=combinedx)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model

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
    global SIMULATING_GAMES_FLAG
    SIMULATING_GAMES_FLAG = True
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
    SIMULATING_GAMES_FLAG = False
    return
    
def train_model(model, model_number = 0, epochs=10, batch_size=4096*2,data_folder="NotRandomDataset_1", validation_sz=10000):
    
    ds, n_files = read_to_dataset([f"{data_folder}/Vectors"],add_channel=False,shuffle_files=True, return_n_files=True)
    # First take validation data without shuffling, to get fully different games
    val_ds = ds.take(validation_sz).batch(batch_size)
    ds = ds.skip(validation_sz).batch(batch_size)
    # Then make a shuffled buffer for training
    ds = ds.shuffle(buffer_size=4*batch_size)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_log",histogram_freq=2)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=1, restore_best_weights=False, start_from_epoch=0)
    history = model.fit(ds,epochs=epochs, initial_epoch=0, validation_data=val_ds, callbacks=[tensorboard_cb, early_stopping_cb])
    model.save(f"model_conv_{model_number}.h5")
    success = os.system(f"./Utilities/convert-to-tflite.py model_conv_{model_number}.h5 --output_file model_conv_{model_number}.tflite")
    if success != 0:
        print("Error converting to tflite. Exiting.")
        sys.exit(1)
    return history

def get_epoch_number_from_files(folder, model_prefix="model_conv_"):
    """ Find the file with the highest epoch number in the folder.
    """
    reg = f"{model_prefix}([0-9]+).tflite"
    files = os.listdir(folder)
    epoch_numbers = []
    for f in files:
        m = re.match(reg,f)
        if m:
            epoch_numbers.append(int(m.group(1)))
    if len(epoch_numbers) == 0:
        return 0
    return max(epoch_numbers)

if __name__ == "__main__":
    #model = get_model((442,))
    #os.system(f"cp -r FullyRandomDataset-V1/ NotRandomDataset_{1}")
    SIMULATE_EPOCH_NUMBER = get_epoch_number_from_files("./")
    print(f"Starting from epoch {SIMULATE_EPOCH_NUMBER}")
    BATCH_SIZE = 4096*2
    DATASET_FOLDER = "NotRandomDataset_1"
    histories_file = "simulate-train-histories.txt"
    model = tf.keras.models.load_model(f"model_conv_{SIMULATE_EPOCH_NUMBER}.h5")
    #model = get_model((442,))
    histories = []
    while SIMULATE_EPOCH_NUMBER < 200:
        # Clear GPU memory
        tf.keras.backend.clear_session()
        simulate_games(f"model_conv_{SIMULATE_EPOCH_NUMBER}.tflite", only_random = False, nrounds=50, ngames=5000, folder=DATASET_FOLDER)
        history = train_model(model, model_number = SIMULATE_EPOCH_NUMBER+1, epochs=10, batch_size=BATCH_SIZE,data_folder=DATASET_FOLDER, validation_sz=10000)
        histories.append(history)
        with open(histories_file,"a") as f:
            f.write(str(history.history))
            f.write("\n")
        SIMULATE_EPOCH_NUMBER += 1
    
    