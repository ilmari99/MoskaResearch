import os
import re
import pandas as pd
CWD = os.getcwd()
PATH = "./Logs/Vectors/"
print("Current working directory: " + CWD)

def check_unique_vectors(path=PATH):
    """ Check that there are no duplicate vectors in the data"""
    with open(path,"r") as f:
        lines = f.readlines()
        uniq_lines = set(lines)
        print(f"Number of unique lines: {len(uniq_lines)}")
    print("Number of lines: " + str(len(lines)))
    print("Percent unique data: " + str(len(uniq_lines)/len(lines)))


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

def balance_data(path=PATH):
    """ Balance the data by removing the extra 1s"""
    ftype = path.split(".")[-1]
    fname = path.split(".")[0]
    if ftype == "csv":
        data = pd.read_csv(path)
        data.to_pickle(fname +".pkl")
        print(f"Converted {path} to pkl")
    elif ftype == "pkl":
        data = pd.read_pickle(path)
    #rows = random.sample(list(range(data.shape[0])),200000)
    #data = data.iloc[rows,:]
    #data.iloc[:,-5:-1] = data.iloc[:,-5:-1].applymap(lambda x: 1 if x == 2 else 0)
    winners = data[data.iloc[:,-1] == 1]
    losers = data[data.iloc[:,-1] == 0]
    winners = winners.sample(n=losers.shape[0])
    print(f"Winners: {winners.shape}")
    print(f"Losers: {losers.shape}")
    data = pd.concat([winners, losers],axis=0)
    data = data.sample(frac=1).reset_index(drop=True)
    data.to_pickle(f"balanced-{fname}.pkl")
    print(f"Saved data to 'balanced-{fname}.pkl'")
    print(data.head())
    print(data.describe())

if __name__ == "__main__":
    balance_data("combined.csv")
    exit()
    check_line_lengths_equal()
    combine_files("combined.csv")
    get_n_losses(CWD + "\\combined.csv")
    check_unique_vectors(CWD + "\\combined.csv")

