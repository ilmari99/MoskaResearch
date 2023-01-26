#!/usr/bin/env python3
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from check_data import create_tf_dataset

def get_optimal_model():
    global INPUT_SHAPE
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer=None))
    model.add(tf.keras.layers.Dropout(rate=0.226))
    model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer=None))
    model.add(tf.keras.layers.Dropout(rate=0.35))
    model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer=None))
    model.add(tf.keras.layers.Dropout(rate=0.35))
    model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer=None))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0015, amsgrad=True),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
            metrics=['accuracy']
    )
    return model


def get_nn_model(channels=None):
    global INPUT_SHAPE
    model = tf.keras.models.Sequential([
    #norm_layer,
        tf.keras.layers.Input(shape=INPUT_SHAPE),
        #tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Dense(600, activation="elu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(600, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(600, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=False),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy']
        )
    return model

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


def load_from_checkpoint(model : tf.keras.models.Sequential, checkpoint_path : str) -> tf.keras.models.Sequential:
    model.load_weights(checkpoint_path,)
    return model

INPUT_SHAPE = (430,)
if __name__ == "__main__":
    all_dataset = create_tf_dataset(["./Data/3ModelsLog/Vectors/", "./Data/1ModelsLog/Vectors/","./Data/NewerLogs400k/Vectors/"],
                                    add_channel=False,
                                    norm=False,
                                    )
    print(all_dataset.take(1).as_numpy_iterator().next())
    #model = load_from_checkpoint(get_nn_model(),'./model-checkpoints/')
    model = get_nn_model()
    VALIDATION_LENGTH = 100000
    TEST_LENGTH = 100000
    BATCH_SIZE = 4096
    tensorboard_log = "tensorboard-log/"
    checkpoint_filepath = './model-checkpoints/'
    
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)#Add shuffle for NN
    
    if os.path.exists("tensorboard-log/"):
        raise Exception("Tensorboard log directory already exists")
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=20, restore_best_weights=True, start_from_epoch=10)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard-log/",histogram_freq=5,profile_batch=(50,100),)
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
    
    model.save("model.h5")