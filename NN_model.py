import tensorflow as tf
from tensorflow import keras
import numpy as np



def prepare_data(data):

    # Creating one-hot encoded labels for the input, task and output layers
    # "None" will be replaced by the actual data
    input_layer_one_hot = tf.one_hot(None, depth = 9)
    task_layer_one_hot = tf.one_hot(None, depth = 5)
    output_layer_one_hot = tf.one_hot(None, depth = 9)




class RNNModel(tf.keras.Model):
    def __init__(self):
        super(RNNModel, self).__init__()

        # Define the input layer
        self.input_layer = tf.keras.layers.Dense(9, activation = "relu")

        # Define the task layer
        self.task_layer = tf.keras.layers.Dense(5, activation = "relu")

        # # Define the recurrent connection in the hidden layer
        self.recurrent_hidden_layer = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(100, activation = "tanh"))

        # Define the output layer
        self.output_layer = tf.keras.layers.Dense(9, activation = "softmax")


    # Define the forward pass of the model
    @tf.function
    def call(self, input, task):

        # Pass the input to the input layer
        input_layer_output = self.input_layer(input)

        # Pass the input to the hidden layer
        input_to_hidden = self.recurrent_hidden_layer(input_layer_output)

        # Pass task to the hiden layer
        task_to_hidden = self.recurrent_hidden_layer(task)

        # Combine both input and task outputs
        combined_hidden = tf.concat([input_to_hidden, task_to_hidden], axis = -1)

        # Pass combined output to the output layer
        combined_output = self.output_layer(combined_hidden)

        # Separate the input and task layers, implement a direct projection of task layer to the output layer
        separate_output = self.output_layer(task)
        
        # Return both outputs
        return combined_output, separate_output

        



# Initialize the model
LSTM_model = RNNModel()

# Initialize CCE loss as the output layer consists of one-hot encoded labels
cce_loss = tf.keras.losses.CategoricalCrossentropy()