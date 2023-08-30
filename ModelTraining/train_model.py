#!/usr/bin/env python3
import os
import random
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import argparse
from ModelTraining.read_to_dataset import create_dataset

def get_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(units=442, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    x = tf.keras.layers.Dense(units=442, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    x = tf.keras.layers.Dense(units=442, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    x = tf.keras.layers.Dense(units=442, activation="relu")(x)
    x = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model")
    parser.add_argument("data_folders", help="A list of folders containing the data")
    parser.add_argument("--input_shape", help="The input shape of the model", default="(442,1)")
    parser.add_argument("--batch_size", help="The batch size", default="2056")
    DATA_FOLDERS = eval(parser.data_folders)
    INPUT_SHAPE = eval(parser.input_shape)
    BATCH_SIZE = int(parser.batch_size)
    print("Data folders: ",DATA_FOLDERS)
    print("Input shape: ",INPUT_SHAPE)
    print("Batch size: ",BATCH_SIZE)
    
    all_dataset = create_dataset(["./Benchmark1/Vectors/","./HumanLogs/Vectors/"],
        add_channel= True if INPUT_SHAPE[-1] == 1 else False,
        shuffle_files=True,
    )
    
    print(all_dataset.take(1).as_numpy_iterator().next()[0].shape)
    model = get_model(INPUT_SHAPE)
    
    print(model.summary())
    
    VALIDATION_LENGTH = 
    TEST_LENGTH = 50
    BATCH_SIZE = 128*2
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
    all_dataset = all_dataset.shuffle(1000)
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    if os.path.exists(tensorboard_log):
        warnings.warn("Tensorboard log directory already exists!")
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=10, restore_best_weights=True, start_from_epoch=5)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log,histogram_freq=5)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    model.fit(x=train_ds, 
              validation_data=validation_ds, 
              epochs=50, 
              callbacks=[early_stopping_cb, tensorboard_cb, model_checkpoint_callback],
              )
    
    model.evaluate(test_ds, verbose=2)
    
    model.save(model_file)
