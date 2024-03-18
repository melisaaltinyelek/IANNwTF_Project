import tensorflow as tf
from tensorflow import keras
import numpy as np


# # Define the LSTM Model
# LSTM_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(input_num_units, activation = "relu"),
#     tf.keras.layers.LSTM(hidden_num_units, activation = "tanh"),
#     tf.keras.layers.Dense(output_num_units, activation = "softmax") 
# ])


class RNNModel(tf.keras.Model):
    def __init__(self, input_num_units, task_num_units, hidden_num_units, output_num_units):
        super(RNNModel, self).__init__()
        self.input_layer = tf.keras.layers.Dense(input_num_units, activation = "relu")
        self.task_layer = tf.keras.layers.Dense(task_num_units, activation = "relu")
        lstm_cell = tf.keras.layers.LSTMCell(units = hidden_num_units)
        self.hidden_layer = tf.keras.layers.RNN(lstm_cell, activation = "tanh", return_sequences = True)
        self.output_layer = tf.keras.layers.Dense(output_num_units, activation = "linear")

    def call(self, inputs):
        x = self.input_layer(inputs)
        task_output = self.task_layer(inputs)
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

