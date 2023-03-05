#!/usr/bin/env python3
import os
import random
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def create_tf_dataset(paths, add_channel=False,get_part="full", max_files = float("inf")) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        try:
            paths = [paths]
        except:
            raise ValueError("Paths should must be a list of strings")
    # Cards on table, vards in hand + players ready, players out, kopled
    misc_parts = list(range(0,5)) + list(range(161,171))
    card_parts = list((n for n in range(0,431) if n not in misc_parts))
    print(f"Number of card parts" + str(len(card_parts)))
    print(f"Misc parts: {misc_parts}")
    file_paths = []
    for path in paths:
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory")
        file_paths += [os.path.join(path, file) for file in os.listdir(path)]
    print("Total number of files: " + str(len(file_paths)))
    file_paths = random.sample(file_paths, min(len(file_paths), max_files))
    print("Shuffled files.")
    dataset = tf.data.TextLineDataset(file_paths)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=", "), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1]), tf.strings.to_number(x[-1])), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print(f"Getting part: {get_part}")
    # Get only the parts we want
    if get_part == "cards":
        dataset = dataset.map(lambda x,y: (tf.gather(x, card_parts), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif get_part == "misc":
        dataset = dataset.map(lambda x,y: (tf.gather(x, misc_parts), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif get_part != "full":
        raise ValueError(f"get_part should be 'cards', 'misc' or 'full', not {get_part}")
    
    # Add a channel dimension
    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def read_to_numpy_array(dataset : tf.data.Dataset) -> np.ndarray:
    """Reads a tf dataset into a numpy array"""
    # Read the dataset into a numpy array
    array = []
    for x,y in dataset:
        array.append(x.numpy())
    array = np.array(array)
    return array



if __name__ == "__main__":
    dataset = create_tf_dataset(["./Benchmark1/Vectors/"], add_channel=False,max_files=1000)
    array = read_to_numpy_array(dataset)
    first_element = array[0]
    print(first_element)
    U, s, V = np.linalg.svd(array, full_matrices=False,)
    print(f"Sum of singular values: {sum(s)}")
    # Create a rank 10 approximation
    rank = 10
    print(f"Sum of singular values for rank {rank}: {sum(s[:rank])}")
    print(f"Proportion of singular values for rank {rank}: {sum(s[:rank])/sum(s)}")
    V = V[:rank, :]
    np.save("V.npy", V)
    
