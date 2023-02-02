#!/usr/bin/env python3
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

class SeparateInput(tf.keras.layers.Layer):
    """ Layer, that separates the input into two tensors, specifically the 'misc' and 'cards' parts of the input."""
    def __init__(self, misc_parts = None, card_parts = None, **kwargs):
        super(SeparateInput, self).__init__(**kwargs)
        if misc_parts is None:
            misc_parts = list(range(0,5)) + list(range(161,171))
            card_parts = list((n for n in range(0,431) if n not in misc_parts))
        self.misc_parts = misc_parts
        self.card_parts = card_parts
        self.misc_shape = (len(misc_parts),)
        self.card_shape = (len(card_parts),)
    
    def call(self, inputs):
        misc = tf.gather(inputs, self.misc_parts, axis=1)
        cards = tf.gather(inputs, self.card_parts, axis=1)
        return misc, cards
    
class CardConv1D(tf.keras.layers.Layer):
    """ Applies a 1D convolution to the cards part of the input, leaving the misc part untouched.
    The cards part is first extracted from the input with the SeparateInput layer.
    The input is expected to be (batch_size, 430,) with no channels
    The output will be flattened to (batch_size, 430 * filters)
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(CardConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.separate = SeparateInput()
        
    def call(self, inputs):
        misc, cards = self.separate(inputs)
        # Add a channel dimension
        cards = tf.expand_dims(cards, axis=-1)
        cards = self.conv(cards)
        output = tf.concat([misc, self.flatten(cards)], axis=1)
        return output


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
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)))
    return dataset


def get_loaded_model(path) -> tf.keras.models.Sequential:
    model = tf.keras.models.load_model(path,compile=True)
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
    model.add(tf.keras.layers.Conv1D(64,3,activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(32,6, activation="linear"))
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

def get_test_model():
    model = tf.keras.models.Sequential()
    model.add(CardConv1D(64,3,activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Dense(32, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

def load_from_checkpoint(model : tf.keras.models.Sequential, checkpoint_path : str) -> tf.keras.models.Sequential:
    model.load_weights(checkpoint_path,)
    return model

INPUT_SHAPE = (430,)
if __name__ == "__main__":
    all_dataset = create_tf_dataset(["./Data/LastLogs/Vectors/"],
                                    add_channel=False,
                                    get_part="misc"
                                    )
    print(all_dataset.take(1).as_numpy_iterator().next())
    #model = load_from_checkpoint(get_nn_model(),'./model-checkpoints/')
    model = get_misc_model()
    VALIDATION_LENGTH = 100000
    TEST_LENGTH = 100000
    BATCH_SIZE = 4096
    tensorboard_log = "tensorboard-log2/"
    checkpoint_filepath = './model-checkpoints2/'
    model_file = "model2.h5"    

    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)#Add shuffle for NN
    
    if os.path.exists(tensorboard_log):
        raise Exception("Tensorboard log directory already exists")
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=20, restore_best_weights=True, start_from_epoch=10)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log,histogram_freq=5,profile_batch=(50,100),)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    
    
    model.fit(x=train_ds, 
              validation_data=validation_ds, 
              epochs=180, 
              callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback],
              )
    
    model.evaluate(test_ds, verbose=2)
    
    model.save(model_file)
