#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split, cross_val_score


def get_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="elu"),
    #tf.keras.layers.Dense(256, activation=None),
    tf.keras.layers.Conv1D(64,kernel_size=3, activation='relu',data_format="channels_last"),
    #tf.keras.layers.MaxPooling1D(5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0),
        metrics=['accuracy']
        )
    
    return model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "combined.pkl"
    else:
        path = sys.argv[1]
    # X,y = read_x_and_y("combined.csv")
    model = get_model()
    ftype = path.split(".")[-1]
    if ftype == "csv":
        data = pd.read_csv(path)
        fname = path.split(".")[0]
        data.to_pickle(fname +".pkl")
        print(f"Converted {path} to pkl")
    elif ftype == "pkl":
        data = pd.read_pickle(path)
    X = np.array(data.iloc[:,:-1])
    y = np.array(data.iloc[:,-1])
    X = X[:,:,np.newaxis]
    y = y[:,np.newaxis]
    #y = np.asarray(y).astype(int).reshape((-1,1))
    #print(X.shape)
    #print(y.shape)
    # y = 1 if not loss, 0 if loss
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42,train_size=0.8)

    model.fit(x_train,y_train, epochs=5,batch_size=512, validation_data=(x_test,y_test))
    #cross_val_score(model, x_train, y_train, cv=5)