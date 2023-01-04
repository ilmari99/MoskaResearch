#!/usr/bin/env python3
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from check_data import create_tf_dataset

def get_optimal_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(425,)))
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
    #norm_layer = tf.keras.layers.Normalization(axis=2)
    #norm_layer.adapt(x_train)
    #print(norm_layer.adapt_mean, norm_layer.adapt_variance)
    model = tf.keras.models.Sequential([
    #norm_layer,
        #norm_layer,
        #tf.keras.layers.Dense(1024, activation="tanh",kernel_regularizer="l2"),
        tf.keras.layers.BatchNormalization(axis=-1,input_shape=(425,)),
        tf.keras.layers.Dense(425, activation="relu"),
        #tf.keras.layers.Dense(425, activation="relu"),
        #tf.keras.layers.Dense(425, activation="relu"),
        #tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Dense(425, activation="relu"),
        #tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Dense(425, activation="relu"),
        #tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        #tf.keras.layers.Dense(425, activation="relu"),
        #tf.keras.layers.Dense(425, activation="relu"),
        #tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007, amsgrad=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        #loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2),
        #loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
        )
    return model

def get_loaded_model() -> tf.keras.models.Sequential:
    model = tf.keras.models.load_model("./Model5-300/model.h5",compile=True)
    return model

if __name__ == "__main__":
    all_dataset = create_tf_dataset("./RandomLogs/Vectors/",add_channel=False)
    model = get_optimal_model()
    VALIDATION_LENGTH = 100000
    TEST_LENGTH = 100000
    BATCH_SIZE = 4096
    SHUFFLE_BUFFER_SIZE = 100000
    print(all_dataset.element_spec)
    
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)#Add shuffle for NN
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=5)
    if os.path.exists("tensorboard-logs/"):
        raise Exception("Tensorboard log directory already exists")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard-logs/")
    
    model.fit(x=train_ds, validation_data=validation_ds, epochs=200, callbacks=[early_stopping_cb, tensorboard_cb])
    
    model.evaluate(test_ds, verbose=1)
    
    model.save("model.h5")