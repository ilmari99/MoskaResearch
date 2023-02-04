#!/usr/bin/env python3
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def create_tf_dataset(paths, add_channel=False,get_part="full") -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        try:
            paths = [paths]
        except:
            raise ValueError("Paths should must be a list of strings")
    # Cards on table, vards in hand + players ready, players out, kopled
    misc_parts = list(range(0,5)) + list(range(161,171))
    card_parts = list((n for n in range(0,431) if n not in misc_parts))
    print(f"Number of card parts" + str(len(card_parts)))
    print(f"Misc parts: {misc_parts}")
    file_paths = []
    for path in paths:
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory")
        file_paths += [os.path.join(path, file) for file in os.listdir(path)]
    print("Number of files: " + str(len(file_paths)))
    random.shuffle(file_paths)
    print("Shuffled files.")
    dataset = tf.data.TextLineDataset(file_paths)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=", "), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1]), tf.strings.to_number(x[-1])), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print(f"Getting part: {get_part}")
    # Get only the parts we want
    if get_part == "cards":
        dataset = dataset.map(lambda x,y: (tf.gather(x, card_parts), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif get_part == "misc":
        dataset = dataset.map(lambda x,y: (tf.gather(x, misc_parts), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif get_part != "full":
        raise ValueError(f"get_part should be 'cards', 'misc' or 'full', not {get_part}")
    
    # Add a channel dimension
    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), y))
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

def get_nn_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.Dense(600, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.35))
    model.add(tf.keras.layers.Dense(550, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.35))
    model.add(tf.keras.layers.Dense(500, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(450, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy']
        )
    return model

def get_card_model(standalone = False, compile_ = True):
    model = tf.keras.models.Sequential()
    if standalone:
        model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.Conv1D(128,3,activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(32,12, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Flatten())
    if standalone:
        model.add(tf.keras.layers.Dense(400,activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.4))
        model.add(tf.keras.layers.Dense(300,activation="relu"))
        model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    if compile_:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
            metrics=['accuracy']
        )
    return model

def get_misc_model(compile_=True):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(12,activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(12,activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(12,activation="relu"))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    if compile_:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
            metrics=['accuracy']
        )
    return model


def get_branched_model():
    misc_parts = list(range(0,5)) + list(range(161,171))
    card_parts = list((n for n in range(0,430) if n not in misc_parts))
    print(f"In total {len(card_parts)} card parts and {len(misc_parts)} misc parts: {len(card_parts)+len(misc_parts)}")
    inputs = tf.keras.layers.Input(shape=(430,))
    card_data = tf.gather(inputs, tf.constant(card_parts, dtype=tf.int32), axis=1)
    card_data = tf.expand_dims(card_data, axis=-1)
    misc_data = tf.gather(inputs,tf.constant(misc_parts, dtype=tf.int32), axis=1)
    
    # Model for card input (415,1); Convolutions can be used for this, since there are patterns that can be found
    img = tf.keras.layers.BatchNormalization(axis=1)(card_data)
    img = tf.keras.layers.Conv1D(128,3,activation="linear")(img)
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Conv1D(64,5,activation="linear")(img)
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Conv1D(16,3,activation="linear")(img)
    img = tf.keras.layers.LeakyReLU(alpha=0.3)(img)
    img = tf.keras.layers.Flatten()(img)
    img = tf.keras.layers.Dense(600,activation="relu")(img)
    img = tf.keras.layers.Dropout(rate=0.5)(img)
    img = tf.keras.layers.Dense(400,activation="relu")(img)
    img = tf.keras.layers.Dropout(rate=0.5)(img)
    img = tf.keras.layers.Dense(200,activation="relu")(img)
    img = tf.keras.layers.Dropout(rate=0.4)(img)
    img = tf.keras.layers.Dense(200,activation="relu")(img)
    # Output a vector, hopefully describing what are the benefits and weaknesses of the known cards (table, others, self)
    img_out = tf.keras.layers.Dense(50,activation="relu")(img)
    
    # Small model for misc input (15,)
    #misc = tf.keras.layers.Dense(15,activation="relu")(misc_data)
    #misc = tf.keras.layers.Dense(12,activation="relu")(misc)
    # The output of this, only really matters, when cards in deck is low
    # This output describes generally, how the position is good/bad
    # Ie. can the player kopl, are they the target, which players are ready
    #misc_out = tf.keras.layers.Dense(5,activation="linear")(misc)
    misc_out = misc_data
    
    # Combine information from misc and card data
    combined = tf.keras.layers.Concatenate(axis=-1)([img_out,misc_out])
    combined = tf.keras.layers.Dense(75,activation="relu")(combined)
    combined = tf.keras.layers.Dropout(rate=0.4)(combined)
    combined = tf.keras.layers.Dense(75,activation="relu")(combined)
    combined = tf.keras.layers.Dropout(rate=0.4)(combined)
    combined = tf.keras.layers.Dense(65,activation="relu")(combined)
    combined_out = tf.keras.layers.Dense(1,activation="sigmoid")(combined)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=combined_out, name="branched_model")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        loss=[tf.keras.losses.BinaryFocalCrossentropy(gamma = 1.5, from_logits=False,label_smoothing=0)],
        metrics=['accuracy'],
        )
    return model

    
    
    

def get_conv_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.Conv1D(128,3,activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
    model.add(tf.keras.layers.Conv1D(32,6, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(32,12, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(500,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(400,activation="relu"))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2,from_logits=False,label_smoothing=0),
        metrics=['accuracy'],
    )
    return model

INPUT_SHAPE = (430,)
if __name__ == "__main__":
    all_dataset = create_tf_dataset(["./LastLogs1/Vectors/", "./LastLogs2/Vectors/","./LastLogs3/Vectors/"],
                                    add_channel=False,
                                    get_part="full"
                                    )
    print(all_dataset.take(1).as_numpy_iterator().next()[0].shape)
    #model = load_from_checkpoint(get_nn_model(),'./model-checkpoints/')
    model = get_branched_model()
    print(model.summary())
    VALIDATION_LENGTH = 500000
    TEST_LENGTH = 500000
    BATCH_SIZE = 4096
    tensorboard_log = "tensorboard-log/"
    checkpoint_filepath = './model-checkpoints/'
    model_file = "model.h5"    

    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)#Add shuffle for NN
    
    if os.path.exists(tensorboard_log):
        raise Exception("Tensorboard log directory already exists")
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=8, restore_best_weights=True, start_from_epoch=1)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log,histogram_freq=5)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    model.fit(x=train_ds, 
              validation_data=validation_ds, 
              epochs=100, 
              callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback],
              )
    
    model.evaluate(test_ds, verbose=2)
    
    model.save(model_file)
