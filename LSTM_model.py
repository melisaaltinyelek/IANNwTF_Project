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

# Read the training and testing datasets
training_dataframe = pd.read_csv("df_training_samples_for_conditioning.csv")
test_dataframe = pd.read_csv("df_testing_samples_for_evaluation.csv")

def create_test_samples(test_dataframe, cue_position = 0, condition = "B"):

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
        df_val_h = test_dataframe.loc[(test_dataframe["cue_position"] == cue_position)]
    else:
        df_val_h = test_dataframe.loc[(test_dataframe["cue_position"] == cue_position) & (test_dataframe["prev_task"] == condition)]

    # Loop through the lists and extract input - output sequences
    test_ds = [literal_eval(x) for x in df_val_h["input"].tolist()]
    test_ds_pred = [literal_eval(x) for x in df_val_h["curr_output"].tolist()]

    return test_ds, test_ds_pred

test_ds, test_ds_pred = create_test_samples(test_dataframe = test_dataframe, cue_position = 9, condition = "B")

# Create dataframes to store accuracies and cue possitions for the tasks A&B and A&C
df_AB_all_accuracies = pd.DataFrame()
df_AC_all_accuracies = pd.DataFrame()
dfAB_accuracy_cue_pos = pd.DataFrame()
dfAC_accuracy_cue_pos = pd.DataFrame()

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None, df_val = test_dataframe):
         
        """
        Performs evaluation on test samples at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Dictionary containing the training logs.
        df_val : pandas.DataFrame
            The dataframe containing the validation data.

        """

        # For condition A&B
        global dfAB_accuracy_cue_pos
        X_AB_test_0, y_AB_test_0 = create_test_samples(df_val, cue_position = 0, condition = "B")
        X_AB_test_1, y_AB_test_1 = create_test_samples(df_val, cue_position = 1, condition = "B")
        X_AB_test_2, y_AB_test_2 = create_test_samples(df_val, cue_position = 2, condition = "B")
        X_AB_test_3, y_AB_test_3 = create_test_samples(df_val, cue_position = 3, condition = "B")
        X_AB_test_4, y_AB_test_4 = create_test_samples(df_val, cue_position = 4, condition = "B")
        X_AB_test_5, y_AB_test_5 = create_test_samples(df_val, cue_position = 5, condition = "B")
        X_AB_test_6, y_AB_test_6 = create_test_samples(df_val, cue_position = 6, condition = "B")
        X_AB_test_7, y_AB_test_7 = create_test_samples(df_val, cue_position = 7, condition = "B")
        X_AB_test_8, y_AB_test_8 = create_test_samples(df_val, cue_position = 8, condition = "B")
        X_AB_test_9, y_AB_test_9 = create_test_samples(df_val, cue_position = 9, condition = "B")

        res_AB_eval_0 = self.model.evaluate(np.array(X_AB_test_0), np.array(y_AB_test_0), verbose = 0)
        res_AB_eval_1 = self.model.evaluate(np.array(X_AB_test_1), np.array(y_AB_test_1), verbose = 0)
        res_AB_eval_2 = self.model.evaluate(np.array(X_AB_test_2), np.array(y_AB_test_2), verbose = 0)
        res_AB_eval_3 = self.model.evaluate(np.array(X_AB_test_3), np.array(y_AB_test_3), verbose = 0)
        res_AB_eval_4 = self.model.evaluate(np.array(X_AB_test_4), np.array(y_AB_test_4), verbose = 0)
        res_AB_eval_5 = self.model.evaluate(np.array(X_AB_test_5), np.array(y_AB_test_5), verbose = 0)
        res_AB_eval_6 = self.model.evaluate(np.array(X_AB_test_6), np.array(y_AB_test_6), verbose = 0)
        res_AB_eval_7 = self.model.evaluate(np.array(X_AB_test_7), np.array(y_AB_test_7), verbose = 0)
        res_AB_eval_8 = self.model.evaluate(np.array(X_AB_test_8), np.array(y_AB_test_8), verbose = 0)
        res_AB_eval_9 = self.model.evaluate(np.array(X_AB_test_9), np.array(y_AB_test_9), verbose = 0)

        if (dfAB_accuracy_cue_pos.empty):
            # Store accuracy in dfAB_accuracy_cue_pos
            dfAB_accuracy_cue_pos.insert(0, "val_acc_cuepos0", [res_AB_eval_0[1]], True)
            dfAB_accuracy_cue_pos.insert(1, "val_acc_cuepos1", [res_AB_eval_1[1]], True)
            dfAB_accuracy_cue_pos.insert(2, "val_acc_cuepos2", [res_AB_eval_2[1]], True)
            dfAB_accuracy_cue_pos.insert(3, "val_acc_cuepos3", [res_AB_eval_3[1]], True)
            dfAB_accuracy_cue_pos.insert(4, "val_acc_cuepos4", [res_AB_eval_4[1]], True)
            dfAB_accuracy_cue_pos.insert(5, "val_acc_cuepos5", [res_AB_eval_5[1]], True)
            dfAB_accuracy_cue_pos.insert(6, "val_acc_cuepos6", [res_AB_eval_6[1]], True)
            dfAB_accuracy_cue_pos.insert(7, "val_acc_cuepos7", [res_AB_eval_7[1]], True)
            dfAB_accuracy_cue_pos.insert(8, "val_acc_cuepos8", [res_AB_eval_8[1]], True)
            dfAB_accuracy_cue_pos.insert(9, "val_acc_cuepos9", [res_AB_eval_9[1]], True)
        else:
            dfAB_temp_accuracy_cue_pos = pd.DataFrame()
            dfAB_temp_accuracy_cue_pos.insert(0, "val_acc_cuepos0", [res_AB_eval_0[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(1, "val_acc_cuepos1", [res_AB_eval_1[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(2, "val_acc_cuepos2", [res_AB_eval_2[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(3, "val_acc_cuepos3", [res_AB_eval_3[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(4, "val_acc_cuepos4", [res_AB_eval_4[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(5, "val_acc_cuepos5", [res_AB_eval_5[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(6, "val_acc_cuepos6", [res_AB_eval_6[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(7, "val_acc_cuepos7", [res_AB_eval_7[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(8, "val_acc_cuepos8", [res_AB_eval_8[1]], True)
            dfAB_temp_accuracy_cue_pos.insert(9, "val_acc_cuepos9", [res_AB_eval_9[1]], True)   
            dfAB_accuracy_cue_pos = pd.concat([dfAB_accuracy_cue_pos, dfAB_temp_accuracy_cue_pos])  
        
        # For condition A&C
        global dfAC_accuracy_cue_pos
        X_AC_test_0, y_AC_test_0 = create_test_samples(df_val, cue_position = 0, condition = "C")
        X_AC_test_1, y_AC_test_1 = create_test_samples(df_val, cue_position = 1, condition = "C")
        X_AC_test_2, y_AC_test_2 = create_test_samples(df_val, cue_position = 2, condition = "C")
        X_AC_test_3, y_AC_test_3 = create_test_samples(df_val, cue_position = 3, condition = "C")
        X_AC_test_4, y_AC_test_4 = create_test_samples(df_val, cue_position = 4, condition = "C")
        X_AC_test_5, y_AC_test_5 = create_test_samples(df_val, cue_position = 5, condition = "C")
        X_AC_test_6, y_AC_test_6 = create_test_samples(df_val, cue_position = 6, condition = "C")
        X_AC_test_7, y_AC_test_7 = create_test_samples(df_val, cue_position = 7, condition = "C")
        X_AC_test_8, y_AC_test_8 = create_test_samples(df_val, cue_position = 8, condition = "C")
        X_AC_test_9, y_AC_test_9 = create_test_samples(df_val, cue_position = 9, condition = "C")

        res_AC_eval_0 = self.model.evaluate(np.array(X_AC_test_0), np.array(y_AC_test_0), verbose = 0)
        res_AC_eval_1 = self.model.evaluate(np.array(X_AC_test_1), np.array(y_AC_test_1), verbose = 0)
        res_AC_eval_2 = self.model.evaluate(np.array(X_AC_test_2), np.array(y_AC_test_2), verbose = 0)
        res_AC_eval_3 = self.model.evaluate(np.array(X_AC_test_3), np.array(y_AC_test_3), verbose = 0)
        res_AC_eval_4 = self.model.evaluate(np.array(X_AC_test_4), np.array(y_AC_test_4), verbose = 0)
        res_AC_eval_5 = self.model.evaluate(np.array(X_AC_test_5), np.array(y_AC_test_5), verbose = 0)
        res_AC_eval_6 = self.model.evaluate(np.array(X_AC_test_6), np.array(y_AC_test_6), verbose = 0)
        res_AC_eval_7 = self.model.evaluate(np.array(X_AC_test_7), np.array(y_AC_test_7), verbose = 0)
        res_AC_eval_8 = self.model.evaluate(np.array(X_AC_test_8), np.array(y_AC_test_8), verbose = 0)
        res_AC_eval_9 = self.model.evaluate(np.array(X_AC_test_9), np.array(y_AC_test_9), verbose = 0)

        if (dfAC_accuracy_cue_pos.empty):
            # Store accuracy in dfAB_accuracy_cue_pos
            dfAC_accuracy_cue_pos.insert(0, "val_acc_cuepos0", [res_AC_eval_0[1]], True)
            dfAC_accuracy_cue_pos.insert(1, "val_acc_cuepos1", [res_AC_eval_1[1]], True)
            dfAC_accuracy_cue_pos.insert(2, "val_acc_cuepos2", [res_AC_eval_2[1]], True)
            dfAC_accuracy_cue_pos.insert(3, "val_acc_cuepos3", [res_AC_eval_3[1]], True)
            dfAC_accuracy_cue_pos.insert(4, "val_acc_cuepos4", [res_AC_eval_4[1]], True)
            dfAC_accuracy_cue_pos.insert(5, "val_acc_cuepos5", [res_AC_eval_5[1]], True)
            dfAC_accuracy_cue_pos.insert(6, "val_acc_cuepos6", [res_AC_eval_6[1]], True)
            dfAC_accuracy_cue_pos.insert(7, "val_acc_cuepos7", [res_AC_eval_7[1]], True)
            dfAC_accuracy_cue_pos.insert(8, "val_acc_cuepos8", [res_AC_eval_8[1]], True)
            dfAC_accuracy_cue_pos.insert(9, "val_acc_cuepos9", [res_AC_eval_9[1]], True)
        else:
            dfAC_temp_accuracy_cue_pos = pd.DataFrame()
            dfAC_temp_accuracy_cue_pos.insert(0, "val_acc_cuepos0", [res_AC_eval_0[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(1, "val_acc_cuepos1", [res_AC_eval_1[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(2, "val_acc_cuepos2", [res_AC_eval_2[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(3, "val_acc_cuepos3", [res_AC_eval_3[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(4, "val_acc_cuepos4", [res_AC_eval_4[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(5, "val_acc_cuepos5", [res_AC_eval_5[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(6, "val_acc_cuepos6", [res_AC_eval_6[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(7, "val_acc_cuepos7", [res_AC_eval_7[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(8, "val_acc_cuepos8", [res_AC_eval_8[1]], True)
            dfAC_temp_accuracy_cue_pos.insert(9, "val_acc_cuepos9", [res_AC_eval_9[1]], True)   
            dfAC_accuracy_cue_pos = pd.concat([dfAC_accuracy_cue_pos, dfAC_temp_accuracy_cue_pos])  
        

def flatten_list(l):

    """
    Flattens a list of lists into a single list.

    Parameters
    ----------
    l : list
        The list of lists to be flattened.

    Returns
    ----------
    list
        A flattened list.
    """
      
    l_flattend =  [x for xs in l for x in xs] 
    return l_flattend

def convert_df(df):

    """
    Converts a dataframe into a flattened dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be converted.

    Returns
    ----------
    pandas.DataFrame
        A flattened dataframe
    """

    col_names = df.columns
    df = pd.DataFrame(np.array(flatten_list(df.values.tolist())).reshape(len(df),len(df.columns)))
    # df = df.rename(columns={0:col_names[0], 1: col_names[1], 2: col_names[2], 3: col_names[3]})
    df = df.rename(columns={0:col_names[0], 1: col_names[1]})
    return df

# Store the losses in a dataframe
df_losses = pd.DataFrame()

helper_index = 0 # Onlly set if you run this cell for the first time
helper_index += 1

# Initialize and store MyCustomCallback() class in my_val_callback 
my_val_callback = MyCustomCallback()

def train_model(training_dataframe, n_neurons, n_batch, desired_mse, learning_rate):

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
        tf.keras.layers.Dense(9, activation = "softmax")
    ])
    
    # Define Adam optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    # Define the loss function (MSE) and optimizer (Adam)
    model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ["accuracy"])  

    # Initialize a list to store training history
    mse_history = []

    # Train the network until the desired MSE has been reached
    while True:
        # Train the model for one epoch
        training_history = model.fit(X_train, y_train,
                                    epochs = 1,
                                    batch_size = n_batch,
                                    verbose = 1,
                                    shuffle = False,
                                    callbacks = [my_val_callback])
        
        # Append the MSE for this epoch to the history
        mse_history.append(training_history.history)
        
        # Check if the MSE has reached the desired threshold
        if training_history.history["loss"][0] < desired_mse:
            print(f"The desired MSE ({desired_mse}) has been reached. The training is over.")
            break

    return mse_history

mse_history = train_model(training_dataframe = training_dataframe, n_neurons = 5, n_batch = 1, desired_mse = 0.001, learning_rate = 0.001)

# Store all results in the dataframe
df_AB_all_accuracies = pd.concat([df_AB_all_accuracies, dfAB_accuracy_cue_pos.reset_index()])
df_AC_all_accuracies = pd.concat([df_AC_all_accuracies, dfAC_accuracy_cue_pos.reset_index()])

#%%

# Convert mse_history dictionary to DataFrame and flatten it
loss_df = pd.DataFrame.from_dict(mse_history)
loss_df = convert_df(loss_df)

# Concatenate the converted loss DataFrame with df_losses
df_losses = pd.concat([df_losses,convert_df(loss_df)])

in_title = "model_" + str(helper_index)

# Plot for condition A&B
print("\n CONDITION A&B: \n")
dfAB_accuracy_cue_pos.reset_index().drop("index", axis = 1).plot()
loss_df["accuracy"].plot()
title_AC = "Condition A&B: " + in_title
plt.title(title_AC)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
print(dfAB_accuracy_cue_pos)

print("\n CONDITION A&C: \n")
dfAC_accuracy_cue_pos.reset_index().drop("index", axis = 1).plot()
loss_df["accuracy"].plot()
title_AC = "Condition A&C: " + in_title 
plt.title(title_AC)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
print(dfAB_accuracy_cue_pos)

#%%

# Extract the number of epochs
epochs_list = [len(history) for history in mse_history]

# Calculate mean and standard deviation 
mean_epochs = np.mean(epochs_list)
std_epochs = np.std(epochs_list)

print(f"The mean number of epochs: {mean_epochs}")
print(f"The standard deviation of epochs: {std_epochs}")
#%% 

df_losses.to_csv("loss_df_run2.csv")
df_AB_all_accuracies.to_csv("df_AB_all_accuracies_run2.csv")
df_AC_all_accuracies.to_csv("df_AC_all_accuracies_run2.csv")

#%%

# Plot the MSE history across epochs
plt.plot(mse_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training MSE History")
plt.show()

#%%

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
trained_model, mse_history = train_model(df = training_dataframe)

# Test the model
test_loss = test_model(trained_model, "df_testing_samples_for_evaluation.csv")

#%%

# Plot the MSE history across epochs
plt.plot(mse_history)
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
