import os
import random
import tensorflow as tf

""" Create a TensorFlow dataset from (multiple) folders of vector files.
"""

MISC_PARTS = list(range(0,5)) + list(range(161,170))
CARD_PARTS = list((n for n in range(0,430) if n not in MISC_PARTS))

def create_dataset(paths, add_channel = False, shuffle_files = True, return_n_files = False) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if os.path.isdir(path)]
    if shuffle_files:
        random.shuffle(file_paths)

    print("Found {} files".format(len(file_paths)))

    dataset = tf.data.TextLineDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    # Split all lines by commas to get a list of strings
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=","), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Convert the list of strings to a list ints and take the last element as the label (0,1)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1], out_type=tf.int32), tf.strings.to_number(x[-1], out_type=tf.int32)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Add a channel dimension if necessary
    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if return_n_files:
        return dataset, len(file_paths)
    return dataset
            





