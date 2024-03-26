# Jenny Arndt, 26.03.2024, hh:mm
#%%
# import libraries
import pandas as pd
from pandas import DataFrame, concat
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt
from ast import literal_eval # for data prepocessing 

#%%
# read and prepare data
df = pd.read_csv("ALLInputOutputSamples_TasksABCDE_withcues0.csv")

stimulus_input = [literal_eval(x) for x in df["stimulus_input"].tolist()]
task_input = [literal_eval(x) for x in df["task_input"].tolist()]
output = [literal_eval(x) for x in df["output"].tolist()]
cue = [[x] for x in df["cue"]]

df = df.drop('output', axis=1)
df.insert(0, "output", output, True)
df = df.drop('task_input', axis=1)
df.insert(0, "task_input", task_input, True)
df = df.drop('stimulus_input', axis=1)
df.insert(0, "stimulus_input", stimulus_input, True)
df = df.drop('cue', axis=1)
df.insert(0, "cue", cue, True)

def flatten_list(l):
    """flatten python list
    
    Parameters
    ----------
    l : list

    Returns
    -------
    flattened_list : list
    """
    flattened_list = [num for elem in l for num in elem]
    return flattened_list

def Cloning(li1, times):
    """clone list n times
    
    Parameters
    ----------
    li1 : list
    times : int

    Returns
    ---------
    l : list (2D)
        list of lists
    """
    li_copy = li1[:]
    l = []
    for i in range(times):
        l.append(li_copy)
    return l

# training variables: cue, stimulus_input, task_input
train_ds = np.array([ Cloning(flatten_list(x),times = 20) for x in df[["cue", "stimulus_input", "task_input"]].values.tolist()])
train_ds_pred = np.array([ Cloning(flatten_list(x),times = 20) for x in df[["output"]].values.tolist()])
#check shape: np.shape(train_ds)

#%%
# configure, design and fit the network
X = train_ds
y = train_ds_pred
n_batch = 1
n_epoch = 22
n_neurons = 10

model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(9))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=1, shuffle=False)
# %%
# plot results
plt.plot(history.history['loss'])
plt.legend()

# %%
