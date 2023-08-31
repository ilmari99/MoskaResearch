#!/usr/bin/env python3
import os
import warnings
import tensorflow as tf
import sys
import ast
import argparse
from read_to_dataset import read_to_dataset

def get_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(600, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    
    x = tf.keras.layers.Dense(550, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    
    x = tf.keras.layers.Dense(450, activation="relu")(x)
    x = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
            loss='binary_crossentropy',
            metrics=['accuracy'])
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model")
    # Datafolders is a list of strings, so we need to evaluate it as a list of strings
    parser.add_argument("data_folders", help="The data folders to train on", default="./FullyRandomDataset-V1/",nargs='?')
    parser.add_argument("--input_shape", help="The input shape of the model", default="(442,)")
    parser.add_argument("--batch_size", help="The batch size", default="4096")
    parser = parser.parse_args()
    DATA_FOLDERS = ["./FullyRandomDataset-V1/Vectors"]
    INPUT_SHAPE = eval(parser.input_shape)
    BATCH_SIZE = int(parser.batch_size)
    print("Data folders: ",DATA_FOLDERS)
    print("Input shape: ",INPUT_SHAPE)
    print("Batch size: ",BATCH_SIZE)
    
    all_dataset, n_files = create_dataset(DATA_FOLDERS,
        add_channel= True if INPUT_SHAPE[-1] == 1 else False,
        shuffle_files=True,
        return_n_files=True,
    )
    
    print(all_dataset.take(1).as_numpy_iterator().next()[0].shape)
    model = get_model(INPUT_SHAPE)
    
    print(model.summary())
    
    approx_num_states = 80 * n_files
    
    VALIDATION_LENGTH = int(0.06 * approx_num_states)
    TEST_LENGTH = int(0.06 * approx_num_states)
    BATCH_SIZE = 4096
    SHUFFLE_BUFFER_SIZE = 4*BATCH_SIZE
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
    all_dataset = all_dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
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
