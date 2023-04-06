#!/usr/bin/env python3
import os
import random
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from create_bitmap_dataset import create_bitmap_dataset


def create_tf_dataset(paths, add_channel=False,sep=",") -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        try:
            paths = [paths]
        except:
            raise ValueError("Paths should must be a list of strings")
    # Cards on table, vards in hand + players ready, players out, kopled
    file_paths = []
    for path in paths:
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory")
        file_paths += [os.path.join(path, file) for file in os.listdir(path)]
    print("Number of files: " + str(len(file_paths)))
    random.shuffle(file_paths)
    print("Shuffled files.")
    dataset = tf.data.TextLineDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=sep), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1]), tf.strings.to_number(x[-1])), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Add a channel dimension
    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def get_loaded_model(path) -> tf.keras.models.Sequential:
    model = tf.keras.models.load_model(path,compile=True)
    return model
    

def load_from_checkpoint(model : tf.keras.models.Sequential, checkpoint_path : str) -> tf.keras.models.Sequential:
    model.load_weights(checkpoint_path,)
    return model
    
def get_transfer_model(base_model_path):
    global INPUT_SHAPE
    base_model = tf.keras.models.load_model(base_model_path,compile=True)
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(431, activation="linear")(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(425, activation="linear")(x)
    output = base_model(x, training=False)
    new_model = tf.keras.Model(inputs, output)
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        metrics = ["accuracy"],
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0)
    )
    return new_model

def _get_nn_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.Dense(400, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(300, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(200, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    #model.add(tf.keras.layers.Dropout(0.4))
    #model.add(tf.keras.layers.Dense(550, activation="relu"))
    #model.add(tf.keras.layers.Dropout(0.3))
    #model.add(tf.keras.layers.Dense(550, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=False),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy']
        )
    return model


def get_nn_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.Dense(600, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(550, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(500, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(450, activation="relu"))
    #model.add(tf.keras.layers.Dropout(0.4))
    #model.add(tf.keras.layers.Dense(550, activation="relu"))
    #model.add(tf.keras.layers.Dropout(0.3))
    #model.add(tf.keras.layers.Dense(550, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=False),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy']
        )
    return model


def get_branched_model():
    global INPUT_SHAPE
    misc_parts = list(range(0,26))
    card_parts = list(range(26,442))
    print(f"In total {len(card_parts)} card parts and {len(misc_parts)} misc parts: {len(card_parts)+len(misc_parts)}")
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    card_data = tf.gather(inputs, tf.constant(card_parts, dtype=tf.int32), axis=1)
    #card_data = tf.keras.layers.BatchNormalization(axis=-1)(card_data)
    card_data = tf.expand_dims(card_data, axis=-1)
    #card_data = tf.reshape(card_data,[8,52,1])
    misc_data = tf.gather(inputs,tf.constant(misc_parts, dtype=tf.int32), axis=1)
    misc_data = tf.repeat(tf.expand_dims(misc_data,axis=1),416,axis=1)
    card_data = tf.concat([card_data,misc_data],axis=-1)
    # Model for card input (415,1); Convolutions can be used for this, since there are patterns that can>
    #img = tf.keras.layers.LocallyConnected1D(6,5,activation="linear",kernel_regularizer="l2")(card_data)
    img = tf.keras.layers.Conv1D(32,4)(card_data) # Combinations of same cards etc.
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Conv1D(32,4)(card_data) # What type of hands, are in the players hands
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Conv1D(32,14)(card_data) # What type of hands, are in the players hands
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Conv1D(32,14)(card_data) # What type of hands, are in the players hands
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Flatten()(img)
    img = tf.keras.layers.Dropout(rate=0.3)(img)
    img = tf.keras.layers.Dense(400,activation="relu")(img)
    img = tf.keras.layers.Dropout(rate=0.4)(img)
    img = tf.keras.layers.Dense(400,activation="relu")(img)
    img = tf.keras.layers.Dropout(rate=0.4)(img)
    img = tf.keras.layers.Dense(400,activation="relu")(img)
    img = tf.keras.layers.Dropout(rate=0.4)(img)
    img = tf.keras.layers.Dense(1,activation="sigmoid")(img)
    # Output a vector, hopefully describing what are the benefits and weaknesses of the known cards (tab>
    #img_out = tf.keras.layers.Dense(30,activation="tanh")(img)
    
    # Small model for misc input (15,)
    #misc = tf.keras.layers.Dense(15,activation="relu")(misc_data)
    #misc = tf.keras.layers.Dense(12,activation="relu")(misc)
    # The output of this, only really matters, when cards in deck is low
    # This output describes generally, how the position is good/bad
    # Ie. can the player kopl, are they the target, which players are ready
    #misc_out = tf.keras.layers.Dense(5,activation="linear")(misc)
    #misc_out = misc_data
    
    # Combine information from misc and card data
    #combined = tf.keras.layers.Concatenate(axis=-1)([img_out,misc_out])
    #combined = tf.keras.layers.Dense(50,activation="relu")(combined)
    #combined = tf.keras.layers.Dropout(rate=0.3)(combined)
    #combined = tf.keras.layers.Dense(25,activation="relu")(combined)
    #combined = tf.keras.layers.Dropout(rate=0.3)(combined)
    #combined = tf.keras.layers.Dense(25,activation="relu")(combined)
    #combined_out = tf.keras.layers.Dense(1,activation="sigmoid")(combined)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=img, name="branched_model")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy'],
        )
    return model


def _get_conv_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.Conv1D(8,4,activation="linear"))
    #model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(16,14, activation="linear"))
    #model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(32,52, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(600,activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.5))
    #model.add(tf.keras.layers.Dense(400,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(600,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(600,activation="relu"))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy'],
    )
    return model

def get_conv_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.Conv1D(4,4,activation="linear", kernel_regularizer="l2"))
    #model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(4,14, activation="linear", kernel_regularizer="l2"))
    #model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(8,52, activation="linear", kernel_regularizer="l2"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(600,activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.5))
    #model.add(tf.keras.layers.Dense(400,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(600,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(600,activation="relu",))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=False),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy'],
    )
    return model

INPUT_SHAPE = (442,)
if __name__ == "__main__":
    #all_dataset = create_tf_dataset(["./Benchmark1/Vectors/", "./Benchmark2/Vectors/", "./Benchmark3/Vectors/", "./HumanLogs/Bitmaps/"],
    #all_dataset = create_bitmap_dataset(["./Dataset-Lumi/Vectors/","./Dataset-Lumi2/Vectors/","./Dataset-Lumi3/Vectors/"],
    all_dataset = create_tf_dataset(["./Dataset-nfmt/Vectors/",
                                    "./Dataset-nfmt-no-rand/Vectors/", "./Dataset-nfmt-no-rand-2/Vectors/", "./Dataset-nfmt-no-rand-3/Vectors/"],
    add_channel=False,
    )
    print(all_dataset.take(1).as_numpy_iterator().next()[0].shape)
    #model = load_from_checkpoint(get_nn_model(),'./model-checkpoints/')
    #model = get_loaded_model("./Model-nn1-BB/model.h5")
    model = get_nn_model()
    print(model.summary())
    VALIDATION_LENGTH = 3000000
    TEST_LENGTH = 1000000
    BATCH_SIZE = 4096*4
    SHUFFLE_PREFETCH_BUF = 4*BATCH_SIZE
    tensorboard_log = "tensorboard-log/"
    checkpoint_filepath = './model-checkpoints/'
    model_file = "model.h5" 
    if len(sys.argv) > 1:
           to_dir = sys.argv[1]
           tensorboard_log = os.path.join(to_dir,tensorboard_log)
           checkpoint_filepath = os.path.join(to_dir,checkpoint_filepath)
           model_file = os.path.join(to_dir,model_file)
    print("Tensorboard log directory: ",tensorboard_log)
    print("Checkpoint directory: ",checkpoint_filepath)
    print("Model file: ",model_file)
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH)
    train_ds = train_ds.shuffle(SHUFFLE_PREFETCH_BUF).batch(BATCH_SIZE)

    if os.path.exists(tensorboard_log):
        warnings.warn("Tensorboard log directory already exists!")
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=12, restore_best_weights=True, start_from_epoch=5)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log,histogram_freq=5)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    model.fit(x=train_ds, 
              validation_data=validation_ds, 
              epochs=150, 
              callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback],
              )
    
    model.evaluate(test_ds, verbose=2)
    
    model.save(model_file)
