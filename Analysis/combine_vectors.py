
""" Combine vectors from all folders in a directory into a single file. """
import os
import sys
from typing import List

def get_vector_paths(dir_path : str) -> List[str]:
    """ Find all folders named 'Vectors' in the given directory and return paths to all files in them. """
    vector_paths : List[str] = []
    for root, dirs, files in os.walk(dir_path):
        if "Vectors" in dirs:
            for file in os.listdir(os.path.join(root, "Vectors")):
                vector_paths.append(os.path.join(root, "Vectors", file))
    return vector_paths

def combine_vectors(dir_path : str, out_path : str):
    """ Combine all vectors in the given directory into a single file. """
    vector_paths = get_vector_paths(dir_path)
    with open(out_path, "w") as outfile:
        for vector_path in vector_paths:
            with open(vector_path, "r") as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write("\n")

if __name__ == "__main__":
    in_file = "./FireBaseLogs"
    out_file = "./vectors.txt"
    if len(sys.argv) == 3:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
    combine_vectors(in_file, out_file)