#!/usr/bin/env python3
import os
import warnings
import tensorflow as tf
import sys
import ast
import argparse
from read_to_dataset import read_to_dataset
import re
import datetime

def get_base_model(input_shape):
    """ Same model used in the original study
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(600, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Dense(550, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    x = tf.keras.layers.Dense(450, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
    combinedx = tf.keras.layers.Dense(100, activation="relu")(combinedx)
    combinedx = tf.keras.layers.Dropout(rate=0.4)(combinedx)
    combinedx = tf.keras.layers.Dense(60, activation="relu")(combinedx)
    combinedx = tf.keras.layers.Dropout(rate=0.4)(combinedx)
    combinedx = tf.keras.layers.Dense(30, activation="relu")(combinedx)
    combinedx = tf.keras.layers.Dense(units=1, activation="sigmoid")(combinedx)
    
    model = tf.keras.Model(inputs=inputs, outputs=combinedx)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model")
    # Datafolders is a list of strings, so we need to evaluate it as a list of strings
    #parser.add_argument("data_folders", help="The data folders to train on", default="./FullyRandomDataset-V1/",nargs='?')
    parser.add_argument("--input_shape", help="The input shape of the model", default="(442,)")
    parser.add_argument("--batch_size", help="The batch size", default="4096")
    parser.add_argument("--model_file", help="The file to save the model to", default="")
    parser = parser.parse_args()
    DATA_FOLDERS = [
        "Datasets/Dataset400kV3Random1",
        #"Datasets/Dataset400kV4Random1",
        #"Datasets/Dataset400kV5Random1",
        #"Datasets/Dataset400kV6Random0",
        #"Datasets/Dataset400kV7Random0",
        #"Datasets/Dataset400kV7-1Random3",
    ]
    #"Vectors" folder
    DATA_FOLDERS = [path + os.sep + "Vectors" for path in DATA_FOLDERS]
    
    INPUT_SHAPE = eval(parser.input_shape)
    BATCH_SIZE = int(parser.batch_size)
    print("Data folders: ",DATA_FOLDERS)
    print("Input shape: ",INPUT_SHAPE)
    print("Batch size: ",BATCH_SIZE)
    from_loaded_model = False
    is_conv = False
    
    
    
    all_dataset, n_files = read_to_dataset(DATA_FOLDERS,
        add_channel = True if INPUT_SHAPE[-1] == 1 else False,
        shuffle_files=True,
        return_n_files=True,
    )
    print(all_dataset.take(1).as_numpy_iterator().next()[0].shape)
    all_dataset = all_dataset.take(10000)
    #model = get_base_model(INPUT_SHAPE)
    #model = get_branched_model(INPUT_SHAPE)
    model = tf.keras.models.load_model("/home/ilmari/python/MoskaResearch/ModelH5Files/model-basic-from-1000k-games-only-NN-games.h5")
    from_loaded_model = True
    #is_conv = True
    
    print(model.summary())
    
    approx_num_states = 80 * n_files
    
    VALIDATION_LENGTH = int(0.07 * approx_num_states)
    TEST_LENGTH = int(0.05 * approx_num_states)
    VALIDATION_LENGTH = 10
    TEST_LENGTH = 10
    print(f"Validation length: {VALIDATION_LENGTH}")
    BATCH_SIZE = 4*4096
    SHUFFLE_BUFFER_SIZE = 2*BATCH_SIZE
    
    # For naming the model file, we need date, dataset versions, conv or not conv, on top of a model or not
    # Find the dataset version number (V1, V2, V3, V7-1, etc)
    reg = re.compile(r"V\d ?-?\d?")
    ds_versions = [reg.findall(folder)[0] for folder in DATA_FOLDERS]
    print("Dataset versions: ",ds_versions)
    #DDMM
    date = datetime.datetime.now().strftime("%d%m")
    
    # DDMM_<type>_<from_model>_<dataset_versions>.h5
    parts = []
    parts.append(date)
    parts.append("conv" if is_conv else "basic")
    parts.append("top" if from_loaded_model else "notop")
    parts.append("".join(ds_versions))
    
        
    default_model_file = "_".join(parts) + ".h5"
    model_file = parser.model_file if parser.model_file != "" else default_model_file
    print("Model file: ",model_file)
    
    
    model_main_name = os.path.splitext(os.path.basename(model_file))[0]
    model_file_path = f"ModelH5Files/{model_file}"
    tensorboard_log = f"Tensorboard-logs/{model_main_name}"
    checkpoint_filepath = f"NNCheckpoints/{model_main_name}"
    print("Tensorboard log directory: ",tensorboard_log)
    print("Checkpoint directory: ",checkpoint_filepath)
    print("Model file: ",model_file)
    if model_file is None:
        raise ValueError("Model file must be specified")
    #all_dataset = all_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    if os.path.exists(tensorboard_log):
        warnings.warn("Tensorboard log directory already exists!")
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=5, restore_best_weights=True, start_from_epoch=0)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log,histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    model.fit(x=train_ds,
              validation_data=validation_ds,
              initial_epoch=0,
              epochs=1, 
              callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback],
              )
    
    result = model.evaluate(test_ds, verbose=2)
    
    # Add the first 4 decimals of test loss to the model file name
    loss = str(round(result[0],4))
    loss = loss[2:]
    model_file = model_file_path[:-3] + f"_{loss}.h5"
    
    # Os change the Tensorboard log and checkpoint directory names to include the loss
    os.rename(tensorboard_log, tensorboard_log + "_" + loss)
    os.rename(checkpoint_filepath, checkpoint_filepath + "_" + loss)
    model.save(model_file)
