#!/usr/bin/env python3
import os
import tensorflow as tf
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert a keras model to tflite")
    parser.add_argument("file_path", help="Path to the keras model", default="model.h5")
    parser.add_argument("--output_file", help="Path to the output file", default=None)
    parser = parser.parse_args()
    output_file = parser.output_file
    file_path = parser.file_path
    if output_file is None:
        output_file = file_path.replace(".h5", ".tflite")
        
    print("Converting '{}' to '{}'".format(file_path, output_file))

    model = tf.keras.models.load_model(file_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_file, "wb") as f:
        f.write(tflite_model)
    print("Done.")


