import tensorflow as tf
from tensorflow import keras
import numpy as np


class RNNModel(tf.keras.Model):
    def __init__(self, input_num_units, task_num_units, hidden_num_units, output_num_units):
        super(RNNModel, self).__init__()
        # Define the input layer
        self.input_layer = tf.keras.layers.Dense(input_num_units, activation = "relu")
        # Define the task layer
        self.task_layer = tf.keras.layers.Dense(task_num_units, activation = "relu")
        # Define the recurrent connection by using LSTMCell
        lstm_cell = tf.keras.layers.LSTMCell(units = hidden_num_units)
        # Pass the LSTMCell to the hidden RNN layer to achieve a recurrent connection
        self.hidden_layer = tf.keras.layers.RNN(lstm_cell, activation = "tanh", return_sequences = True)
        # Define the output layer
        self.output_layer = tf.keras.layers.Dense(output_num_units, activation = "softmax")

    def call(self, input, task):
        x = self.input_layer(input)
        task_output = self.task_layer(task)
        x = tf.concat([x, task_output], axis = 1)
        x = self.hidden_layer(x)
        output = self.output_layer(x)
        return output


# Define the number of units for each layer
input_num_units = 9
task_num_units = 5
hidden_num_units = 100
output_num_units = 9




# Create an instance of the RNNModel
LSTM_model = RNNModel(input_num_units, task_num_units, hidden_num_units, output_num_units)

# Creating one-hot encoded labels for the input, task and output layers
# "None" will be replaced by the actual data
input_layer_one_hot = tf.one_hot(None, depth = input_num_units)
task_layer_one_hot = tf.one_hot(None, depth = task_num_units)
output_layer_one_hot = tf.one_hot(None, depth = output_num_units)