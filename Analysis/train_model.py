#!/usr/bin/env python3
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_forest_model():
    model = RandomForestClassifier(n_estimators=600, max_depth=10,random_state=42)
    return model

def get_nn_model(x_train):
    #norm_layer = tf.keras.layers.Normalization(axis=2)
    #norm_layer.adapt(x_train)
    #print(norm_layer.adapt_mean, norm_layer.adapt_variance)
    model = tf.keras.models.Sequential([
        #norm_layer,
        tf.keras.layers.Dense(1024, activation="tanh"),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512, activation="linear",),
        tf.keras.layers.Dense(512, activation="relu",),
        #tf.keras.layers.Dense(256, activation=None),
        #tf.keras.layers.Conv1D(512,kernel_size=5, activation='relu',data_format="channels_last",padding="causal"),
        #tf.keras.layers.MaxPooling1D(5),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="linear"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True,),
        #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
        )
    
    return model



if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "test-data-bigger.pkl"
    else:
        path = sys.argv[1]
    # X,y = read_x_and_y("combined.csv")
    ftype = path.split(".")[-1]
    if ftype == "csv":
        data = pd.read_csv(path)
        fname = path.split(".")[0]
        #data.to_pickle(fname +".pkl",)
        #print(f"Converted {path} to pkl")
    elif ftype == "pkl":
        data = pd.read_pickle(path)
    #rows = random.sample(list(range(data.shape[0])),200000)
    #data.iloc[rows,:].to_csv("test-data-bigger.csv",index=False,header=False)
    data.iloc[:,-5:-1] = data.iloc[:,-5:-1].applymap(lambda x: 1 if x == 2 else 0)
    print(data.describe())
    X = np.array(data.iloc[:,:-1])
    # Convert to one-hot
    y = np.array(data.iloc[:,-1])
    #y = np.asarray(y).astype(int).reshape((-1,1))
    print(X.shape)
    print(y.shape)

    # y = 1 if not loss, 0 if loss
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.2)
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    #y_train = np.asarray(y_train).astype('float32').reshape((1,-1))
    #y_test = np.asarray(y_test).astype('float32').reshape((1,-1))
    y_train = y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))
    #print(x_train.shape)
    #print(y_train.shape)
    #print(y_train)
    model = get_nn_model(x_train)
    model.fit(x_train,y_train, epochs=10,batch_size=512, validation_data=(x_test,y_test))
    #cross_val_score(model, x_train, y_train, cv=5)