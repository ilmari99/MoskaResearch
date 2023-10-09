import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy.stats import multivariate_normal
from sklearn.metrics import log_loss
#import polars as pl

def show_distributions(df, cols = None):
    if cols is None:
        cols = df.columns
    figs = []
    axes = []
    for col in cols:
        fig, ax = plt.subplots()
        figs.append(fig)
        axes.append(ax)
        # Different colors for 'Didn't lose' column
        ax.hist(df[df["Didn't lose"] == 1][col], bins=100, density=True, alpha=0.5, label="Didn't lose")
        ax.hist(df[df["Didn't lose"] == 0][col], bins=100, density=True, alpha=0.5, label="Lost")
        ax.set_title(col)
    return figs, axes
        

if __name__ == "__main__":
    #main_df = pl.read_csv("combined.csv", has_header=False, separator=",", new_columns=list(mapper.values()))
    main_df = pd.read_csv("combined.csv",header=None,sep=",")
    # set headers according to 'vector_index_labels.json'
    mapper = json.load(open("vector_index_labels.json","r"))
    # Keys to int
    mapper = {int(k):v for k,v in mapper.items()}
    
    # Rename columns
    main_df = main_df.rename(columns=mapper)
    
    # Plot the distributions of the data
    # First 10 columns
    df = main_df.iloc[:,120:140]
    df = pd.concat([df, main_df["Didn't lose"]], axis=1)
    show_distributions(df)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
        
    