# Latest Update: 29.03.2024
#%%
import pandas as pd
from pandas import DataFrame, concat
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM
from keras.models import load_model
from matplotlib import pyplot as plt
from ast import literal_eval
import h5py

#%%
# Read the data
#df = pd.read_csv("ALLInputOutputSamples_TasksABCDE_withcues0.csv") # test_data
df = pd.read_csv("df_training_samples_for_conditioning.csv")
df_val = pd.read_csv("df_testing_samples_for_evaluation.csv")
# Loop through the lists that are represented as strings in the original dataframe
# and convert them to actual lists.
train_ds = [literal_eval(x) for x in df["input"].tolist()]
train_ds_pred = [literal_eval(x) for x in df["curr_output"].tolist()]

# test_ds = [literal_eval(x) for x in df_val["input"].tolist()]
# test_ds_pred = [literal_eval(x) for x in df_val["curr_output"].tolist()]
#%%
def create_test_samples(df_val, cue_position = 0, condition = "B"):
    if cue_position > 9:
        print(f"cue_position is greater than 9! Given is: {cue_position}")
        raise ValueError
    if not((condition == "C") | (condition =="B") | (condition == "both")):
        print(f"condition is neither 'C' nor 'B! Given is: {condition}")
        raise ValueError
    df_val_h = None
    if condition == "both":
        df_val_h = df_val.loc[(df_val["cue_position"] == cue_position)]
    else:
        df_val_h = df_val.loc[(df_val["cue_position"] == cue_position) & (df_val["prev_task"] == condition)]
    test_ds = [literal_eval(x) for x in df_val_h["input"].tolist()]
    test_ds_pred = [literal_eval(x) for x in df_val_h["curr_output"].tolist()]
    return test_ds, test_ds_pred

test_ds, test_ds_pred = create_test_samples(df_val = df_val, cue_position = 9, condition = "B")

# Check the shape of the training dataset:
# np.shape(train_ds) -> (X,20,15)
# np.shape(train_ds_pred) -> (X,9)
df_all_accuracies = pd.DataFrame()

# callback
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback  & https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras
df_accuracy_cue_pos = pd.DataFrame()
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, df_val = df_val, condition = "C"):
        global df_accuracy_cue_pos
        X_test_0, y_test_0 = create_test_samples(df_val, cue_position = 0, condition = condition)
        X_test_1, y_test_1 = create_test_samples(df_val, cue_position = 1, condition = condition)
        X_test_2, y_test_2 = create_test_samples(df_val, cue_position = 2, condition = condition)
        X_test_3, y_test_3 = create_test_samples(df_val, cue_position = 3, condition = condition)
        X_test_4, y_test_4 = create_test_samples(df_val, cue_position = 4, condition = condition)
        X_test_5, y_test_5 = create_test_samples(df_val, cue_position = 5, condition = condition)
        X_test_6, y_test_6 = create_test_samples(df_val, cue_position = 6, condition = condition)
        X_test_7, y_test_7 = create_test_samples(df_val, cue_position = 7, condition = condition)
        X_test_8, y_test_8 = create_test_samples(df_val, cue_position = 8, condition = condition)
        X_test_9, y_test_9 = create_test_samples(df_val, cue_position = 9, condition = condition)

        res_eval_0 = self.model.evaluate(np.array(X_test_0), np.array(y_test_0), verbose = 0)
        res_eval_1 = self.model.evaluate(np.array(X_test_1), np.array(y_test_1), verbose = 0)
        res_eval_2 = self.model.evaluate(np.array(X_test_2), np.array(y_test_2), verbose = 0)
        res_eval_3 = self.model.evaluate(np.array(X_test_3), np.array(y_test_3), verbose = 0)
        res_eval_4 = self.model.evaluate(np.array(X_test_4), np.array(y_test_4), verbose = 0)
        res_eval_5 = self.model.evaluate(np.array(X_test_5), np.array(y_test_5), verbose = 0)
        res_eval_6 = self.model.evaluate(np.array(X_test_6), np.array(y_test_6), verbose = 0)
        res_eval_7 = self.model.evaluate(np.array(X_test_7), np.array(y_test_7), verbose = 0)
        res_eval_8 = self.model.evaluate(np.array(X_test_8), np.array(y_test_8), verbose = 0)
        res_eval_9 = self.model.evaluate(np.array(X_test_9), np.array(y_test_9), verbose = 0)

        if (df_accuracy_cue_pos.empty):
            # store accuracy in df_accuracy_cue_pos
            df_accuracy_cue_pos.insert(0, "val_acc_cuepos0", [res_eval_0[1]], True)
            df_accuracy_cue_pos.insert(1, "val_acc_cuepos1", [res_eval_1[1]], True)
            df_accuracy_cue_pos.insert(2, "val_acc_cuepos2", [res_eval_2[1]], True)
            df_accuracy_cue_pos.insert(3, "val_acc_cuepos3", [res_eval_3[1]], True)
            df_accuracy_cue_pos.insert(4, "val_acc_cuepos4", [res_eval_4[1]], True)
            df_accuracy_cue_pos.insert(5, "val_acc_cuepos5", [res_eval_5[1]], True)
            df_accuracy_cue_pos.insert(6, "val_acc_cuepos6", [res_eval_6[1]], True)
            df_accuracy_cue_pos.insert(7, "val_acc_cuepos7", [res_eval_7[1]], True)
            df_accuracy_cue_pos.insert(8, "val_acc_cuepos8", [res_eval_8[1]], True)
            df_accuracy_cue_pos.insert(9, "val_acc_cuepos9", [res_eval_9[1]], True)
        else:
            df_temp_accuracy_cue_pos = pd.DataFrame()
            df_temp_accuracy_cue_pos.insert(0, "val_acc_cuepos0", [res_eval_0[1]], True)
            df_temp_accuracy_cue_pos.insert(1, "val_acc_cuepos1", [res_eval_1[1]], True)
            df_temp_accuracy_cue_pos.insert(2, "val_acc_cuepos2", [res_eval_2[1]], True)
            df_temp_accuracy_cue_pos.insert(3, "val_acc_cuepos3", [res_eval_3[1]], True)
            df_temp_accuracy_cue_pos.insert(4, "val_acc_cuepos4", [res_eval_4[1]], True)
            df_temp_accuracy_cue_pos.insert(5, "val_acc_cuepos5", [res_eval_5[1]], True)
            df_temp_accuracy_cue_pos.insert(6, "val_acc_cuepos6", [res_eval_6[1]], True)
            df_temp_accuracy_cue_pos.insert(7, "val_acc_cuepos7", [res_eval_7[1]], True)
            df_temp_accuracy_cue_pos.insert(8, "val_acc_cuepos8", [res_eval_8[1]], True)
            df_temp_accuracy_cue_pos.insert(9, "val_acc_cuepos9", [res_eval_9[1]], True)   
            df_accuracy_cue_pos = pd.concat([df_accuracy_cue_pos, df_temp_accuracy_cue_pos])  

# plot losses and accuracies
def flatten_list(l):
    l_flattend =  [x for xs in l for x in xs] 
    return l_flattend
def convert_df(df):
    col_names = df.columns
    df = pd.DataFrame(np.array(flatten_list(df.values.tolist())).reshape(len(df),len(df.columns)))
    # df = df.rename(columns={0:col_names[0], 1: col_names[1], 2: col_names[2], 3: col_names[3]})
    df = df.rename(columns={0:col_names[0], 1: col_names[1]})
    return df
#%%
X = np.array(train_ds)
y = np.array(train_ds_pred)

X_val = np.array(test_ds)
y_val = np.array(test_ds_pred)

# Define the number of batches, epochs and neurons
n_batch = 1
n_epoch = 20
n_neurons = 5 #100 keep it as low as possible - such that the network is capable to solve the task (and is stable for same hyperparameters) (if it is to large, the network too complex the task is too easy and the few epochs needed until desired_mse)
my_val_callback = MyCustomCallback()
df_accuracy_cue_pos = pd.DataFrame()

# Initialize the network and add an LSTM layer
model = Sequential()
model.add(LSTM(n_neurons, input_shape = (X.shape[1], X.shape[2]), stateful = False))
model.add(Dense(9, activation='softmax'))
model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ['accuracy'])

# Define the desired MSE threshold as in the original paper
desired_mse = 0.001

# Initialize a list to store training history
mse_history = []

# Train the network until the desired MSE has been reached
while True:
    # Train the model for one epoch
    training_history = model.fit(X, y, 
                   epochs = 1, 
                   batch_size = n_batch, 
                   verbose = 1, 
                   shuffle = False, # shuffle happens on the batch_axis not time_axis but since our batch is 1 it is not shuffling anything
                   callbacks=[my_val_callback]) 
    
    # Append the MSE for this epoch to the history
    mse_history.append(training_history.history)
    
    # Check if the MSE has reached the desired threshold
    if training_history.history["loss"][0] < desired_mse:
        print(f"The desired MSE ({desired_mse}) has been reached. The training is over.")
        break

# store all results in df
df_all_accuracies = pd.concat([df_all_accuracies, df_accuracy_cue_pos.reset_index()]) # not reset_index()

# plot results
loss_df = pd.DataFrame.from_dict(mse_history)
df_accuracy_cue_pos.reset_index().drop("index", axis = 1).plot()
convert_df(loss_df)["accuracy"].plot()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
df_accuracy_cue_pos
#%%
# Plot the MSE history across epochs
plt.plot(mse_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training MSE History")
plt.show()


