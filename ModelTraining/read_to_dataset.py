import os
import random
import tensorflow as tf
from tqdm import tqdm
import multiprocessing as mp
import argparse

""" Create a TensorFlow dataset from (multiple) folders of vector files.
"""

MISC_PARTS = list(range(0,5)) + list(range(161,170))
CARD_PARTS = list((n for n in range(0,430) if n not in MISC_PARTS))

def read_to_dataset(paths, add_channel=False, shuffle_files=True) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if os.path.isdir(path)]
    if shuffle_files:
        random.shuffle(file_paths)

    print("Found {} files".format(len(file_paths)))
    def txt_line_to_tensor(x):
        s = tf.strings.split(x, sep=",")
        s = tf.strings.to_number(s, out_type=tf.float32)
        return (s[:-1], s[-1])

    def ds_maker(x):
        ds = tf.data.TextLineDataset(x, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        ds = ds.map(txt_line_to_tensor,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False)
        return ds
    
    
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.interleave(ds_maker,
                                cycle_length=tf.data.experimental.AUTOTUNE,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                deterministic=False)

    # Add a channel dimension if necessary
    if add_channel:
        dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset, len(file_paths)




