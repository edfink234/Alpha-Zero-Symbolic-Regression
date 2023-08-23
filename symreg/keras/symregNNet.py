import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

"""
NeuralNet for the game of Symbolic Regression.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class symregNNet():
    def __init__(self, game, args):
        # game params
        self.board = game.getBoardSize()  # a number
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board, 1))  # s: batch_size x board x 1

        # Adjust the convolutional layers for 1D data
        h_conv1 = Activation('relu')(BatchNormalization(axis=-1)(Conv1D(args.num_channels, 3, padding='same')(self.input_boards)))  # batch_size x board x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=-1)(Conv1D(args.num_channels, 3, padding='same')(h_conv1)))  # batch_size x board x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=-1)(Conv1D(args.num_channels, 3, padding='same')(h_conv2)))  # batch_size x board x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=-1)(Conv1D(args.num_channels, 3, padding='valid')(h_conv3)))  # batch_size x (board-2) x num_channels

        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
        
        self.model.summary()

'''
The given code defines a neural network (NNet) class called `symregNNet` designed for solving the problem of symbolic regression using TensorFlow and Keras. Symbolic regression is a task in which the goal is to find a mathematical expression that best fits a given set of data points.

Let's break down the code step by step:

1. Importing Dependencies:
   The code starts by importing necessary libraries and modules, including `sys` for system-specific functionality, custom utility functions from a file named "utils," and relevant components from TensorFlow and Keras.

2. Class Definition:
   The `symregNNet` class is defined to encapsulate the neural network architecture and training for symbolic regression.

3. Initialization:
   The constructor (`__init__`) takes two arguments: `game` and `args`. `game` represents the symbolic regression problem, and `args` holds various hyperparameters and settings.

4. Gathering Game Parameters:
   The constructor initializes some attributes based on the provided `game` object:
   - `self.board`: Represents the size of the input board (a symbolic regression dataset).
   - `self.action_size`: Represents the number of possible actions (output size).

5. Building the Neural Network Architecture:
   The neural network architecture is built using convolutional and fully connected layers. The code constructs a 1D convolutional neural network (CNN) to process the 1D symbolic regression data.

   - `self.input_boards`: Represents the input layer, accepting data of shape `(self.board, 1)`.

   The input data goes through a series of convolutional layers (`Conv1D`), each followed by batch normalization and ReLU activation. The convolutions aim to capture local patterns in the input data.

6. Adjusting Convolutional Layers:
   The convolutional layers are adjusted for 1D data:
   - Four convolutional layers (`h_conv1` to `h_conv4`) are created with increasing depths. The last convolutional layer uses padding='valid', which reduces the output size.

7. Building the Rest of the Network:
   The comment suggests that there is more code for building the rest of the network. However, this part is missing from the provided code snippet.

8. Flattening and Fully Connected Layers:
   The convolutional output is flattened, and a series of fully connected layers are added to the architecture. Each fully connected layer is followed by batch normalization, ReLU activation, and dropout for regularization.

9. Output Layers:
   - `self.pi`: Represents the output policy layer, which predicts a probability distribution over possible actions using a softmax activation function.
   - `self.v`: Represents the output value layer, which predicts a scalar value representing the estimated value of the input state using a tanh activation function.

10. Model Compilation:
    The model is compiled using the Keras `compile` method. It uses a combination of loss functions and an optimizer:
   - The `categorical_crossentropy` loss is used for the policy output.
   - The `mean_squared_error` loss is used for the value output.
   - The `Adam` optimizer with the specified learning rate (`args.lr`) is used for training.

11. Summary:
    The architecture summary can be obtained using `self.model.summary()`.

It's important to note that some parts of the code are indicated by comments (e.g., "Continue building the rest of the network" and "At the end, define your model and compile it"). These parts are not provided in the code snippet and would need to be completed for the network to be fully functional.
'''