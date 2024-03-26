import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from ast import literal_eval


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

# training variables: cue, stimulus_input, task_input
train_ds = [flatten_list(x) for x in df[["cue", "stimulus_input", "task_input"]].values.tolist()]
# train_ds = tf.expand_dims(train_ds, axis=-1)
train_ds_pred = [flatten_list(x) for x in df[["output"]].values.tolist()]


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(15, activation = "sigmoid"))
model.add(tf.keras.layers.LSTM(100, activation = "tanh"))
model.add(tf.keras.layers.Dense(9, activation = "softmax"))

model.compile(loss='mean_squared_error', optimizer='adam')


model.fit(train_ds, epochs = 10)

    # stimulus_data = data["stimulus_input"]
    # task_data = data["task_input"]
    # cue_data = data["cue"]

    # # Define chunk size that represents the iteration/timesteps over the LSTM layer
    # chunk_size = 10

    # # Initialize stimulus and task lists to store chunks of data
    # stimulus_chunks = []
    # task_chunks = []
    # cue_chunks = []

    # # # Loop through the entire stimulus data starting from 0 to 10 (chunk_size)
    # # # slice the stimulus and task chunks based on the defined chunk_size and append them to the lists
    # for i in range(0, len(stimulus_data), chunk_size):
    #     stimulus_chunk = stimulus_data[i:i + chunk_size]
    #     task_chunk = task_data[i:i + chunk_size]
    #     stimulus_chunks.append(stimulus_chunk)
    #     task_chunks.append(task_chunk)


# class RNNModel(tf.keras.Model):
#     def __init__(self):
#         super(RNNModel, self).__init__()

#         # Define the input layer
#         self.input_layer = tf.keras.layers.Dense(9, activation = "sigmoid")

#         # Define the task layer
#         self.task_layer = tf.keras.layers.Dense(5, activation = "sigmoid")

#         # Define the cue layer
#         self.cue_layer = tf.keras.layers.Dense(1, activation = "sigmoid")

#         # # Define the recurrent connection in the hidden layer
#         self.recurrent_layer = tf.keras.layers.LSTM(100, activation = "tanh")

#         # Define the output layer
#         self.output_layer = tf.keras.layers.Dense(9, activation = "softmax")


# class RNNModel(tf.keras.Model):
#     def __init__(self):
#         super(RNNModel, self).__init()
#         self.model = self.build_model()

#     def build_model(self):
#         model = tf.keras.Sequential([
#             tf.keras.layers.Dense(9, activation = "sigmoid"), 
#             tf.keras.layers.Dense(5, activation = "sigmoid"),
#             tf.keras.layers.Dense(1, activation = "sigmoid") 
#             tf.keras.layers.LSTM(100, activation = "tanh", return_sequences = True),
#             tf.keras.layers.Dense(9, activation = "softmax") 
#         ]) 
        
#         return model



    # # Define the forward pass of the model
    # @tf.function
    # def call(self, stimulus, task, cue):

    #     # Pass the input and task to the input layer as it consists of two partitions
    #     combined_input = tf.concat([stimulus, task], axis = -1)

    #     # Pass the combined output to the input layer
    #     input_layer_output = self.input_layer(combined_input)

    #     # Pass the input to the recurrent layer
    #     input_to_recurrent = self.recurrent_hidden_layer(input_layer_output)

    #     # Pass the task to the recurrent layer
    #     task_to_recurrent = self.recurrent_layer(task) 

    #     # Pass the cue to the recurrent layer
    #     cue_to_recurrent = self.recurrent_layer(cue)

    #     # Combine both input, task and cue outputs
    #     combined_recurrent = tf.concat([input_to_recurrent, task_to_recurrent, cue_to_recurrent], axis = -1)

    #     # Pass combined output to the output layer
    #     combined_output = self.output_layer(combined_recurrent)

    #     # Separate the input and task layers, implement a direct projection of task layer to the output layer
    #     separate_output = self.output_layer(task)
        
    #     # Return both outputs
    #     return combined_output, separate_output

        
# # Train the model   
# def train_RNN_model(RNN_model, train_dataset, test_dataset, loss_function, optimizer, num_epochs):
#     None

    

# # Initialize the model
# RNN_model = RNNModel()

# # Initialize MSE loss
# mse_loss = tf.keras.losses.MeanSquaredError()

# # Initialize the SGD optimizer
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.3)
