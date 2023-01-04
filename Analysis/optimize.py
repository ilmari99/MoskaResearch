#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from check_data import create_tf_dataset

def opt_data():
    all_dataset = create_tf_dataset("./BigLogs/Logs/Vectors/",add_channel=False)
    VALIDATION_LENGTH = 200000
    TEST_LENGTH = 200000
    BATCH_SIZE = 4096
    SHUFFLE_BUFFER_SIZE = 100000
    
    validation_ds = all_dataset.take(VALIDATION_LENGTH).batch(BATCH_SIZE)
    
    test_ds = all_dataset.skip(VALIDATION_LENGTH).take(TEST_LENGTH).batch(BATCH_SIZE)
    
    train_ds = all_dataset.skip(VALIDATION_LENGTH+TEST_LENGTH).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)#Add shuffle for NN
    
    return train_ds, validation_ds, test_ds

def get_model(train_ds, validation_ds, test_ds):
        #os.system("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/")
        #os.system("export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11/")
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=425, activation={{choice(["relu","linear"])}},kernel_regularizer={{choice([None,"l2"])}},input_shape=(425,)))
        model.add(tf.keras.layers.Dense(units=425, activation={{choice(["elu","linear",""])}}))
        model.add(tf.keras.layers.Dropout(rate={{uniform(0, 0.5)}}))
        model.add(tf.keras.layers.Dense(units=425, activation={{choice(["relu","linear"])}},kernel_regularizer={{choice([None,"l2"])}}))
        #model.add(tf.keras.layers.Dense(units=425, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dropout(rate={{uniform(0, 0.5)}}))
        model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer={{choice([None,"l2"])}}))
        #model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer=None))
        model.add(tf.keras.layers.Dropout(rate={{uniform(0, 0.5)}}))
        model.add(tf.keras.layers.Dense(units=425, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer=None))
        layers_add = {{choice([0,1,2])}}
        if layers_add > 0:
            for i in range(layers_add):
                model.add(tf.keras.layers.Dense(units=425, activation="relu",kernel_regularizer=None))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid")),
        

        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate={{uniform(0.0001, 0.1)}}),
            loss='binary_crossentropy',
            metrics=['accuracy']
            )
        early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.01,patience=4)
        model.fit(
            train_ds, 
            epochs=20,
            verbose=0,
            validation_data=validation_ds,
            callbacks=[early_stop])
        print(model.summary())
        score, acc = model.evaluate(test_ds, verbose=0)
        print('Test accuracy:', acc)
        print("Test loss: ",score)
        return {'loss': score, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":
    # Verify installation
    print("Tensorflow version: ", tf.__version__)
    print("Python version: ", sys.version)
    #Restrict to last gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #x_train, y_train, x_test, y_test = opt_data()
    best_run, best_model = optim.minimize(model=get_model,
                                            data=opt_data,
                                            algo=tpe.suggest,
                                            max_evals=40,
                                            trials=Trials(),)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)