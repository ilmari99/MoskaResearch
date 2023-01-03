#!/usr/bin/env python3
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from check_data import create_tf_dataset

def get_forest_model():
    model = RandomForestClassifier(n_estimators=600, max_depth=10,random_state=42)
    return model

def get_nn_model():
    #norm_layer = tf.keras.layers.Normalization(axis=2)
    #norm_layer.adapt(x_train)
    #print(norm_layer.adapt_mean, norm_layer.adapt_variance)
    model = tf.keras.models.Sequential([
    #norm_layer,
        #norm_layer,
        #tf.keras.layers.Dense(1024, activation="tanh",kernel_regularizer="l2"),
        tf.keras.layers.Dense(64, activation="relu",kernel_regularizer=None, input_shape=(425,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        #loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
        )
    return model

if __name__ == "__main__":
    all_dataset = create_tf_dataset("./Logs/Vectors/")
    model = get_nn_model()
    VALIDATION_LENGTH = 50000
    TEST_LENGTH = 50000
    BATCH_SIZE = 2*4096
    SHUFFLE_BUFFER_SIZE = 10000
    
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    
    model.fit(train_ds, epochs=10, validation_data=validation_ds,verbose=1)
    
    model.evaluate(test_ds, verbose=1)