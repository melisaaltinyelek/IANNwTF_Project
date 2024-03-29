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

def train_model(training_dataset, n_neurons = 10, n_batch = 1, desired_mse = 0.001, learning_rate = 0.001):

    """ 
    Trains a neural network model using the given training dataset.

    Parameters
    ----------
    training_dataset : str
        The file path or CSV file containing the training dataset.
    n_neurons : int
        The number of neurons in the LSTM layer.
    n_batch : int
        The batch size used during the training.
    desired_mse : int
        The threshold MSE value at which the training stops.

    Returns
    ----------
    model : tf.keras.Sequential()
        The trained neural network model.
    mse_history:
        The list that contains all MSE values stored at each epoch.
    """

    # Read and store the training data
    training_dataframe = pd.read_csv(training_dataset)

    # Loop through the lists that are represented as strings in the original dataframe
    # and convert them to actual lists.
    X_train = [literal_eval(x) for x in training_dataframe["input"].tolist()]
    y_train = [literal_eval(x) for x in training_dataframe["curr_output"].tolist()]

    # Convert the training lists to arrays to access individual elements in the network
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Initialize the network, add an LSTM and a Dense layer
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(n_neurons, input_shape = (X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(9)
    ])
    
    # Define Adam optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    # Define the loss function (MSE) and optimizer (Adam)
    model.compile(loss = "mean_squared_error", optimizer = optimizer)  

    # Initialize a list to store training history
    mse_history = []

    # Train the network until the desired MSE has been reached
    while True:
        # Train the model for one epoch
        training_history = model.fit(X_train, y_train, epochs = 1, batch_size = n_batch, verbose = 1, shuffle = False)
        
        # Append the MSE for this epoch to the history
        mse_history.append(training_history.history["loss"][0])
        
        # Check if the MSE has reached the desired threshold
        if mse_history[-1] < desired_mse:
            print(f"The desired MSE ({desired_mse}) has been reached. The training is over.")
            break

    return model, mse_history

#%%

def create_test_samples(df_val, cue_position, condition = "C"):

    """
    Creates test samples based on the given dataframe, cue position, and condition.

    Parameters
    ----------
    df_val : pandas.DataFrame
        The DataFrame containing the validation data.
    cue_position : int
        The cue position for which test samples are to be created.
    condition : str, optional
        The condition specifying the type of test samples.

    Returns
    ----------
    test_ds : list
        A list of input sequences for the test samples.
    test_ds_pred : list
        A list of expected output sequences for the test samples.

    Raises
    ----------
    ValueError
        If the cue_position is greater than 9 or if the condition is neither "C", "B", nor "both".
    """

    # Check if the cue position is valid
    if cue_position > 9:
        print(f"Cue position is greater than 9! Given is: {cue_position}")
        raise ValueError
    
    # Check if the condition is valid
    if not((condition == "C") | (condition == "B") | (condition == "both")):
        print(f"Condition is neither 'C' nor 'B! Given is: {condition}")
        raise ValueError
    
    df_val_h = None

    # Check and filter the dataframe based on cue position and condition
    if condition == "both":
        df_val_h = df_val.loc[(df_val["cue_position"] == cue_position)]
    else:
        df_val_h = df_val.loc[(df_val["cue_position"] == cue_position) & (df_val["prev_task"] == condition)]

    # Loop through the lists and extract input - output sequences
    test_ds = [literal_eval(x) for x in df_val_h["input"].tolist()]
    test_ds_pred = [literal_eval(x) for x in df_val_h["curr_output"].tolist()]

    return test_ds, test_ds_pred

def test_model(model, test_dataset): 

    """ 
    Evaluates and tests the trained model using the provided test dataset.

    Parameters
    ----------
    model : tf.keras.Sequential
        The trained model to be tested.
    test_dataset : str
        The file path of the CSV file containing the test dataset.

    Returns
    ----------
    test_loss : float
        The loss value obtained by evaluating the model on the test dataset.
    """ 
     
    # Read and store the test data 
    testing_dataframe = pd.read_csv(test_dataset) 
    
    # Create test samples by using create_test_samples()
    X_test, y_test =  create_test_samples(df_val = testing_dataframe, cue_position = 9, condition = "C")
    
    # Convert test samples to arrays
    X_test = np.array(X_test) 
    y_test = np.array(y_test) 
 
    test_loss = model.evaluate(X_test, y_test, verbose = 1) 
 
    print(f"Test Loss: {test_loss}") 
 
    return test_loss

#%%

# Train the model
trained_model, training_history = train_model("df_training_samples_for_conditioning.csv")

# Test the model
test_loss = test_model(trained_model, "df_testing_samples_for_evaluation.csv")

#%%

# Plot the MSE history across epochs
plt.plot(training_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training MSE History")
plt.show()

# plt.plot(training_history.history['loss'])
# plt.legend()

#%%

def freeze_weights(model, filename):

    """
    Freezes model weights after training and saves them to file.

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

freeze_weights(trained_model, "frozen_model.h5")

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

    """
    Load the entire model with its frozen weights.

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
loaded_model.summary()
