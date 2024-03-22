import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


def prepare_data(data):

    # Load the dataset
    data = pd.read_csv(data_as_csv)

    stimulus_data = data["COLUMN_NAME_FOR_STIMULUS"]
    task_data = data["COLUMN_NAME_FOR_TASK"]

    return stimulus_data, task_data


    # Creating one-hot encoded labels for the input, task and output layers
    # "None" will be replaced by the actual data
    # input_layer_one_hot = tf.one_hot(None, depth = 9)
    # task_layer_one_hot = tf.one_hot(None, depth = 5)
    # output_layer_one_hot = tf.one_hot(None, depth = 9)
    # cue_layer_one_hot = tf.one_hot(None, depth = 1)


class RNNModel(tf.keras.Model):
    def __init__(self):
        super(RNNModel, self).__init__()

        # Define the input layer
        self.input_layer = tf.keras.layers.Dense(9, activation = "sigmoid")

        # Define the task layer
        self.task_layer = tf.keras.layers.Dense(5, activation = "sigmoid")

        # Define the cue layer
        self.cue_layer = tf.keras.layers.Dense(1, activation = "sigmoid")

        # # Define the recurrent connection in the hidden layer
        self.recurrent_layer = tf.keras.layers.LSTM(100, activation = "tanh")

        # Define the output layer
        self.output_layer = tf.keras.layers.Dense(9, activation = "softmax")


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



    # Define the forward pass of the model
    @tf.function
    def call(self, stimulus, task, cue):

        # Pass the input and task to the input layer as it consists of two partitions
        combined_input = tf.concat([stimulus, task], axis = -1)

        # Pass the combined output to the input layer
        input_layer_output = self.input_layer(combined_input)

        # Pass the input to the recurrent layer
        input_to_recurrent = self.recurrent_hidden_layer(input_layer_output)

        # Pass the task to the recurrent layer
        task_to_recurrent = self.recurrent_layer(task) 

        # Pass the cue to the recurrent layer
        cue_to_recurrent = self.recurrent_layer(cue)

        # Combine both input, task and cue outputs
        combined_recurrent = tf.concat([input_to_recurrent, task_to_recurrent, cue_to_recurrent], axis = -1)

        # Pass combined output to the output layer
        combined_output = self.output_layer(combined_recurrent)

        # Separate the input and task layers, implement a direct projection of task layer to the output layer
        separate_output = self.output_layer(task)
        
        # Return both outputs
        return combined_output, separate_output

        
# Train the model   
def train_RNN_model(RNN_model, train_dataset, test_dataset, loss_function, optimizer, num_epochs):
    None



# Define the dataset path
data_as_csv = "PATH_TO_DATA"
stimulus_data, task_data = prepare_data(data_as_csv)

# Initialize the model
RNN_model = RNNModel()


# Initialize MSE loss
mse_loss = tf.keras.losses.MeanSquaredError()