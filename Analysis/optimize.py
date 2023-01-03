#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def opt_data():
    data = pd.read_pickle("balanced-test-data-bigger.pkl")
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.astype(float)
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1])
    #y = np.asarray(y).astype(int).reshape((-1,1))
    #print(X.shape)
    #print(y.shape)
    # y = 1 if not loss, 0 if loss
    x_train, x_test,y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.2)
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    y_train = y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))
    return x_train, y_train, x_test, y_test

def get_model(x_train, y_train, x_test, y_test):
        model = tf.keras.Sequential()
        #tf.keras.layers.Dense(1024, activation="tanh",kernel_regularizer="l2"),
        model.add(tf.keras.layers.Dense({{choice([128,256,512,1024,2048])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        if {{choice(["conv1", "noconv"])}} == "conv1":
            model.add(tf.keras.layers.Conv1D({{choice([128,256,512])}},kernel_size={{choice([3,5])}}, activation={{choice(["relu","linear"])}},data_format="channels_last",padding="same"))
        model.add(tf.keras.layers.Dense({{choice([128,256,512,1024,2048])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dropout({{uniform(0, 0.5)}}))
        model.add(tf.keras.layers.Dense({{choice([64,128,256,512,1024])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        if {{choice(["conv2", "noconv"])}} == "conv2":
            model.add(tf.keras.layers.Conv1D({{choice([128,256,512])}},kernel_size={{choice([3,5])}}, activation={{choice(["relu","linear"])}},data_format="channels_last",padding="same"))
        model.add(tf.keras.layers.Dense({{choice([64,128,256,512,1024])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dropout({{uniform(0, 0.5)}}))
        model.add(tf.keras.layers.Dense({{choice([32,64,128,256,512])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dense({{choice([32,64,128,256,512])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dropout({{uniform(0, 0.5)}}))
        model.add(tf.keras.layers.Dense({{choice([16,32,64,128,256,512])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dense({{choice([16,32,64,128,256,512])}}, activation={{choice(["relu","linear"])}},kernel_regularizer=None))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid")),

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate={{uniform(0.0001, 0.1)}}), loss={{choice(["binary_crossentropy","mse"])}}, metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=30, batch_size={{choice([512,1024,2048])}}, verbose=0, validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy:', acc)
        return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":
    # Verify installation
    print("Tensorflow version: ", tf.__version__)
    print("Python version: ", sys.version)
    x_train, y_train, x_test, y_test = opt_data()
    best_run, best_model = optim.minimize(model=get_model,
                                            data=opt_data,
                                            algo=tpe.suggest,
                                            max_evals=100,
                                            trials=Trials(),)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)