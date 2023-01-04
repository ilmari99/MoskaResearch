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
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(425, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        #loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2),
        #loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
        )
    return model

if __name__ == "__main__":
    all_dataset = create_tf_dataset("./BigLogs/Logs/Vectors/",add_channel=False)
    model = get_nn_model()
    VALIDATION_LENGTH = 200000
    TEST_LENGTH = 200000
    BATCH_SIZE = 4096
    SHUFFLE_BUFFER_SIZE = 100000
    
    print(all_dataset.element_spec)
    
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)#Add shuffle for NN
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=5)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard-logs/log1")
    
    model.fit(x=train_ds, validation_data=validation_ds, epochs=100, callbacks=[early_stopping_cb, tensorboard_cb])
    
    model.evaluate(test_ds, verbose=1)
    
    model.save("model.h5")