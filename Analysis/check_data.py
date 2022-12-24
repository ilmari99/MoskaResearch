import os
import re
CWD = os.getcwd()
PATH = "./Logs/Vectors/"
print("Current working directory: " + CWD)

def check_unique_vectors(path=PATH):
    """ Check that there are no duplicate vectors in the data"""
    with open(path,"r") as f:
        lines = f.readlines()
        uniq_lines = set(lines)
        print(f"Number of unique lines: {len(uniq_lines)}") 


def check_line_lengths_equal(path=PATH):
    line_length = -1
    out = True
    for file in os.listdir(path):
        with open(path+file,"r") as f:
            for line in f:
                if line_length == -1:
                    line_length = line.count(",")
                    print("Line length: " + str(line_length))
                if line.count(",") != line_length:
                    print(f"Line length mismatch in {file}")
                    out = False
                    break
    return out

def get_n_losses(path=PATH):
    """ Get the number of losses in each file"""
    with open(path,"r") as f:
        n_losses = 0
        data_length = 0
        for line in f:
            data_length += 1
            line = line.strip()
            if line[-1] == "0":
                n_losses += 1
        print(f"{path} : {n_losses}")
        print(f"Data length: {data_length}")
        print(f"Loss ratio: {n_losses/data_length}")


def combine_files(output,path=PATH):
    """ Combine all files in path into one file"""
    with open(output,"w") as f:
        for file in os.listdir(path):
            with open(path+file,"r") as f2:
                for line in f2:
                    f.write(line)
            f.write("\n")


check_line_lengths_equal()
combine_files("combined.csv")
get_n_losses(CWD + "\\combined.csv")
check_unique_vectors(CWD + "\\combined.csv")

