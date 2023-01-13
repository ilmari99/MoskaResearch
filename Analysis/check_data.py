#!/usr/bin/env python3
import os
import random
import re
import pandas as pd
import tensorflow as tf

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
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=", "))
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1]), tf.strings.to_number(x[-1])))
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
    for file in os.listdir(path):
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
    count = 0
    for i,file in enumerate(duplicate_files):
        os.remove(folder_two + file)
        if i % 10 == 0:
            print(f"{i/len(duplicate_files)*100}% complete")
    return

if __name__ == "__main__":
    CWD = os.getcwd()
    PATH = "./Logs/Vectors/"
    print("Current working directory: " + CWD)
    #find_duplicate_files()
    #balance_data()
    check_data(PATH, count_unique=True)
    #combine_files(PATH,output="combined.csv")
    #get_n_losses("combined.csv")
    #check_unique_vectors("combined.csv")

