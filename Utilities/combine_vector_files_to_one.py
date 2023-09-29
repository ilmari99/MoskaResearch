import os
import argparse

def combine_files(path,output="combined.csv",number_of_files=10000):
    """ Combine all files in path into one file"""
    with open(output,"w") as f:
        for nth_file, file in enumerate(os.listdir(path)):
            if nth_file == number_of_files:
                break
            with open(path+file,"r") as f2:
                for line in f2:
                    f.write(line)
            f.write("\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Combine all files in a folder into one file")
    parser.add_argument("path",help="Path to folder containing files to combine")
    parser.add_argument("--output",help="Name of output file",default="combined.csv")
    parser.add_argument("--number_of_files",help="Number of files to combine",default=10000)
    args = parser.parse_args()
    combine_files(args.path,args.output,args.number_of_files)