import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os

##input_df = the cleaned dataframe from analysis.py removing all 0s and NaAN
## n= number of clusters
## algo = a string containing the algo you wanna use for nnj, should be one of the following 'auto', 'ball_tree', 'kd_tree', 'brute'

def NNJ(input_df,n,algo):
    df = input_df[['centroid-0','centroid-1']]
    X = df.to_numpy()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm=algo).fit(X)
    out = nbrs.kneighbors_graph(X).toarray()
    return out