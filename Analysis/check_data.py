#!/usr/bin/env python3
import os
import random
import re
import pandas as pd
import tensorflow as tf
import sys

def create_tf_dataset(paths, norm=False, add_channel=False) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        try:
            paths = [paths]
        except:
            raise ValueError("Paths should must be a list of strings")
    file_paths = []
    for path in paths:
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory")
        file_paths += [os.path.join(path, file) for file in os.listdir(path)]
    print("Number of files: " + str(len(file_paths)))
    random.shuffle(file_paths)
    print("Shuffled files.")
    dataset = tf.data.TextLineDataset(file_paths)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=", "), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1]), tf.strings.to_number(x[-1])), num_parallel_calls=6)
    if norm:
        dataset = dataset.map(lambda x,y : (tf.divide(x,51),y))
    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)))
    return dataset



def check_data(path, count_unique=False):
    line_length = -1
    out = True
    nlines = 0
    nlosses = 0
    uniques = set()
    files = os.listdir(path)
    for i,file in enumerate(files):
        if i % 1000:
            print(f"Checking file: {i/len(files)*100}% complete")
        with open(path+file,"r") as f:
            for line in f:
                nlines += 1
                if count_unique:
                    uniques.add(line)
                if line_length == -1:
                    line_length = line.count(",")
                    print("Elements: " + str(line_length+1))
                line = line.strip()
                if line[-1] == "0":
                    nlosses += 1
                if line.count(",") != line_length:
                    print(f"Line length mismatch in {file}")
                    out = False
                    break
    if out:
        print(f"All {nlines} lines are the same length")
    if count_unique:
        print(f"Unique lines: {len(uniques)}, {(len(uniques)/nlines) * 100} %")
    print(f"Loss ratio: {nlosses/nlines}")
    return out


def combine_files(path,output="combined.csv"):
    """ Combine all files in path into one file"""
    with open(output,"w") as f:
        for file in os.listdir(path):
            with open(path+file,"r") as f2:
                for line in f2:
                    f.write(line)
            f.write("\n")

    
def find_duplicate_files(remove=False):
    folder_one = "./MB1Logs/Logs/Vectors/"
    folder_two = "./Logs/Vectors/"
    files_in_one = set(os.listdir(folder_one))
    files_in_two = set(os.listdir(folder_two))
    duplicate_files = files_in_one.intersection(files_in_two)
    print(f"Number of duplicate files: {len(duplicate_files)}")
    if not remove:
        return
    for i,file in enumerate(duplicate_files):
        os.remove(folder_two + file)
        if i % 10 == 0:
            print(f"{i/len(duplicate_files)*100}% complete")
    return

if __name__ == "__main__":
    PATH = "./Logs2/Vectors/"
    CWD = os.getcwd()
    if len(sys.argv) > 1:
        PATH = sys.argv[1]
    l = ["./Data/NewerLogs-Inc-ModelBot/Vectors/",
                                     "./Data/NewerLogs60k/Vectors/",
                                     "./Logs/Vectors/",
                                     "./Logs2/Vectors"],
    print("Current working directory: " + CWD)
    print("Checking data for path: " + PATH)
    check_data(PATH, count_unique=True)

