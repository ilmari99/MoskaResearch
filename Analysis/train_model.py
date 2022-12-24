import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split, cross_val_score

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.Conv1D(256, 5, activation='relu'),
    #tf.keras.layers.MaxPooling1D(5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

def read_x_and_y(path):
    X = []
    y = []
# Load data
    counter = 0
    with open(path, 'r') as data:
        line_len = -1
        line = data.readline()
        while line:
            counter += 1
            line = line.strip()
            values = line.split(', ')
            if line_len == -1:
                line_len = len(values)
            #print(line)
            elif len(values) != line_len:
                print("Line length mismatch, Expected: " + str(line_len) + " Got: " + str(len(values)))
                print(line)
                exit()
            if counter % 1000 == 0:
                print(f"Line {counter} read")
            values = [float(x) for x in values]
            y_val = values.pop(-1)
            X.append(values)
            y.append(y_val)
            line = data.readline()
    return X,y

def convert_line_to_list(line):
    print(line)
    line = line.strip()
    values = line.split(', ')
    values = [int(x) for x in values]
    return values

def read_dataset(path):
    arr = np.fromfile(path,int,sep=", ")
    print(arr)
    arr = arr.reshape((-1,422))
    print(arr)
    y = arr[:,-1]
    X = arr[:,:-1]
    return X,y

if __name__ == "__main__":
    path = sys.argv[1]
    # X,y = read_x_and_y("combined.csv")
    data = pd.read_csv(path)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    # y = 1 if not loss, 0 if loss
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42,train_size=0.8)

    model.fit(x_train,y_train, epochs=5,batch_size=128, validation_data=(x_test,y_test))
    #cross_val_score(model, x_train, y_train, cv=5)