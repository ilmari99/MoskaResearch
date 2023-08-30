#!/usr/bin/env python3
import os
import argparse

def verify_equal_length_and_balance(path):
    """
    Check that each vector in a folder of files is the same length
    """
    line_length = -1
    out = True
    nlines = 0
    nlosses = 0
    files = os.listdir(path)
    for i,file in enumerate(files):
        if i % 1000:
            print(f"Checking file: {i/len(files)*100}% complete")
        with open(path+file,"r") as f:
            for line in f:
                nlines += 1
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
    print(f"Loss ratio: {nlosses/nlines}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verify that all vectors in all files in a folder have the same length")
    parser.add_argument("path",help="Path to folder containing files to check")
    args = parser.parse_args()
    verify_equal_length_and_balance(args.path)
    

