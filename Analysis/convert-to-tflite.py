#!/usr/bin/env python3
import os
import tensorflow as tf

file_path = "/home/ilmari/python/moska/ModelMB2/model.h5"
output_file = None

if output_file is None:
    output_file = os.path.splitext(file_path)[0] + ".tflite"
print("Converting '{}' to '{}'".format(file_path, output_file))

model = tf.keras.models.load_model(file_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("./ModelMB2/model.tflite", "wb") as f:
    f.write(tflite_model)


