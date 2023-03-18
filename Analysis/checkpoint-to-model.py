import tensorflow as tf
import os
import sys


def get_model():
    global INPUT_SHAPE
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))
    model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.Conv1D(8,4,activation="linear"))
    #model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(16,14, activation="linear"))
    #model.add(tf.keras.layers.BatchNormalization(axis=1,))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Conv1D(32,52, activation="linear"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(1200,activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.5))
    #model.add(tf.keras.layers.Dense(400,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(800,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(800,activation="relu"))
    model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
    return model

INPUT_SHAPE = (422,1)
if __name__ == "__main__":
    checkpoint_path = "./Model-full-conv3/model-checkpoints/"
    output = os.path.splitext(checkpoint_path)[0] + ".tflite"
    model = get_model()
    model = model.load_weights(checkpoint_path)
    model.save(output)
    os.system("python3 convert-to-tflite.py \"{}\"".format(output))
