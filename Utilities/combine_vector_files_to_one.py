import os
import argparse

def combine_files(path,output="combined.csv"):
    """ Combine all files in path into one file"""
    with open(output,"w") as f:
        for file in os.listdir(path):
            with open(path+file,"r") as f2:
                for line in f2:
                    f.write(line)
            f.write("\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Combine all files in a folder into one file")
    parser.add_argument("path",help="Path to folder containing files to combine")
    parser.add_argument("--output",help="Name of output file",default="combined.csv")
    args = parser.parse_args()
    combine_files(args.path,args.output)