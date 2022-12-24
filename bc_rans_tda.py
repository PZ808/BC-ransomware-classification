RAND_STATE  = 42
SPLIT_RATIO = 0.23
SAVE_TEST_TRAIN = False
DS_FRAC = .12
NF = 7

from heist_tools import *

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import kmapper as km
from kmapper import jupyter
import umap
import sklearn
import sklearn.manifold as manifold

import warnings


warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)
_ = np.seterr(over='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

dft = pd.read_csv('data/dft_dow.csv').sort_values(by='n_day') # read the data with an added column for day


# removing the prefix on some of the labels
study_origins = ["princeton","padua","montreal"] #
for r in study_origins:
    dft['label']=dft['label'].str.replace(r,"")
del study_origins,r

t0 = 2450 # current time
lag = 480 # lag time

# we start by windowing the dataframe and transforming the data in the time window
df_past = window(dft,t0,lag)
print(df_past.head())

txn_features = ['length', 'weight', 'count', 'neighbors', 'looped']
address_features = ['address', 'income']
time_features = ['day_of_week', 'day', 'year', 'n_day']


df_past = col_transform(
    df_past,
    txn_features+[address_features[1]],
    [time_features[0]])


mapper = km.KeplerMapper(verbose=1)

num_dirty = (df_past['label']=='white').value_counts().iloc[1]
print("number of dirty addresses is {}".format(num_dirty))

df_past_graph = df_past[['length', 'weight', 'count', 'neighbors', 'looped','income']]


graph = mapper.map(
    lens=df_past.length,
    X=df_past_graph,
    clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=10))

#print(graph['nodes'])
cluster_data = pd.DataFrame(mapper.data_from_cluster_id('cube0_cluster10', graph, df_past.to_numpy()))
print(cluster_data.shape)