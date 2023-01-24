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
    model = tf.keras.models.Sequential([
    #norm_layer,
        tf.keras.layers.Input(shape=(430,)),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Dense(600, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(550, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(450, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015, amsgrad=False),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        #loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2),
        #loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
        )
    
    return model

def get_loaded_model(path) -> tf.keras.models.Sequential:
    model = tf.keras.models.load_model(path,compile=True)
    return model
    
    
def get_transfer_model():
    base_model = tf.keras.models.load_model("./Model5-300/model.h5",compile=True)
    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=(425,))
    x = tf.keras.layers.Dense(425, activation="relu")(inputs)
    output = base_model(x, training=False)
    new_model = tf.keras.Model(inputs, output)
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
        metrics = ["accuracy"],
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0)
    )
    return new_model

def load_from_checkpoint(model : tf.keras.models.Sequential, checkpoint_path : str) -> tf.keras.models.Sequential:
    model.load_weights(checkpoint_path)
    return model

if __name__ == "__main__":
    all_dataset = create_tf_dataset(["./3ModelsLog/Vectors/","./Logs/Vectors/", "./1ModelsLog/Vectors/","NewerLogs400k/Vectors/" ],add_channel=False, norm=False)
    #mirrored = tf.distribute.MirroredStrategy()
    #with mirrored.scope():
    model = get_nn_model()
    #model = get_loaded_model("./ModelMB10-160/model.h5")
    #model = get_nn_model()
    #model = load_from_checkpoint(model,"./model-checkpoint/")
    VALIDATION_LENGTH = 100000
    TEST_LENGTH = 100000
    BATCH_SIZE = 4*4096
    #SHUFFLE_BUFFER_SIZE = 100000
    
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)#Add shuffle for NN
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=20, restore_best_weights=True, start_from_epoch=10)
    if os.path.exists("tensorboard-log/"):
        raise Exception("Tensorboard log directory already exists")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard-log/", histogram_freq=1, profile_batch=(50,150),write_steps_per_second="epoch")

    checkpoint_filepath = './model-checkpoint/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=False)
    
    model.fit(x=train_ds, validation_data=validation_ds, epochs=250, callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback])
    
    model.evaluate(test_ds, verbose=1)
    
    model.save("model.h5")
