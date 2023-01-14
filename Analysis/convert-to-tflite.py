#!/usr/bin/env python3
import os
import tensorflow as tf
import sys

file_path = "/home/ilmari/python/moska/Models/ModelMB4/model.h5"
output_file = None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if output_file is None:
        output_file = os.path.splitext(file_path)[0] + ".tflite"
    print("Converting '{}' to '{}'".format(file_path, output_file))

    model = tf.keras.models.load_model(file_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_file, "wb") as f:
        f.write(tflite_model)


