#%%
import pandas as pd
from pandas import DataFrame, concat
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM
from keras.models import load_model
from matplotlib import pyplot as plt
from ast import literal_eval
import h5py

#%%

# Read and store the data
df = pd.read_csv("ALLInputOutputSamples_TasksABCDE_withcues0.csv")

# Loop through the lists that are represented as strings in the original dataframe
# and convert them to actual lists.
stimulus_input = [literal_eval(x) for x in df["stimulus_input"].tolist()]
task_input = [literal_eval(x) for x in df["task_input"].tolist()]
output = [literal_eval(x) for x in df["output"].tolist()]
cue = [[x] for x in df["cue"]]

# Drop the existing columns ("cue", "task_input", "stimulus_input", "output") and reinsert the updated values

df = df.drop('cue', axis=1)
df.insert(0, "cue", cue, True)

df = df.drop('task_input', axis=1)
df.insert(0, "task_input", task_input, True)

df = df.drop('stimulus_input', axis=1)
df.insert(0, "stimulus_input", stimulus_input, True)

df = df.drop('output', axis=1)
df.insert(0, "output", output, True)


def flatten_list(list_to_be_flattened):
    """ Takes a nested input list and concatenates them into a single list.
    
    Parameters
    ----------
    list_to_be_flattened : list

    Returns
    ----------
    flattened_list : list
    """

    flattened_list = [num for elem in list_to_be_flattened for num in elem]
    return flattened_list

def clone_list(list_to_be_cloned, times):
    """ Clone list n times.
    
    Parameters
    ----------
    list_1 : list
    times : int

    Returns
    ----------
    list_2D : list (2D)
        list of lists
    """

    list_copy = list_to_be_cloned[:]

    list_2D = []
    for i in range(times):
        list_2D.append(list_copy)
    return list_2D

# Create training datasets: cue, task_input, stimulus_input, output
train_ds = np.array([clone_list(flatten_list(x), times = 20) for x in df[["cue", "task_input", "stimulus_input"]].values.tolist()])
train_ds_pred = np.array([clone_list(flatten_list(x), times = 1) for x in df[["output"]].values.tolist()])
# Check the shape of the training dataset:
# np.shape(train_ds)

#%%

X = train_ds
y = train_ds_pred

# Define the number of batches, epochs and neurons
n_batch = 1
n_epoch = 20
n_neurons = 10

# Initialize the network and add an LSTM layer
model = Sequential()
model.add(LSTM(n_neurons, input_shape = (X.shape[1], X.shape[2]), stateful = False))
model.add(Dense(9))
model.compile(loss = "mean_squared_error", optimizer = "adam")

# Train the network
# training_history = model.fit(X, y, epochs = n_epoch, batch_size = n_batch, verbose = 1, shuffle = False)

# Define the desired MSE threshold as in the original paper
desired_mse = 0.001

# Initialize a list to store training history
mse_history = []

# Train the network until the desired MSE has been reached
while True:
    # Train the model for one epoch
    training_history = model.fit(X, y, epochs = 1, batch_size = n_batch, verbose = 1, shuffle = False)
    
    # Append the MSE for this epoch to the history
    mse_history.append(training_history.history["loss"][0])
    
    # Check if the MSE has reached the desired threshold
    if mse_history[-1] < desired_mse:
        print(f"The desired MSE ({desired_mse}) has been reached. The training is over.")
        break
        
#%%

# Plot the MSE history across epochs
plt.plot(mse_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training MSE History")
plt.show()

# plt.plot(training_history.history['loss'])
# plt.legend()

#%%

def freeze_weights(model, filename):
    """ Freezes model weights after training and saves them to file.

    Parameters
    ----------
    model : keras.models.Sequential()
        The trained model to freeze.

    filename : str
        The filename to save the model.
    
    """
    
    for layer in model.layers:
        layer.trainable = False

    model.save(filename)
    print(f"The model is saved to {filename} with its frozen weights.")

freeze_weights(model, "frozen_model.h5")

#%%

file_path = "frozen_model.h5"

with h5py.File(file_path, 'r') as f:
    # Print the structure of the frozen_model.h5 file
    print("File structure:")
    print(list(f.values())) 

    # Print the weights by iterating over each layer
    for layer_name in f.keys():
        print("\nLayer:", layer_name)
        layer_group = f[layer_name]
        for weight_name in layer_group.keys():
            print("Weight:", weight_name)
            weight_data = np.array(layer_group[weight_name])
            print("Shape:", weight_data.shape)
            print("Data:")
            print(weight_data)

#%%

def load_model_with_frozen_weights(filename):
    """ Load the entire model with its frozen weights.

    Parameters
    ----------
    filename : str
        The filename to load the model from.

    Returns
    ----------
    model : keras.models.Sequential
        The loaded model with frozen weights.
    """

    model = load_model(filename)
    return model

#%%

loaded_model = load_model_with_frozen_weights("frozen_model.h5")
# loaded_model.summary()

