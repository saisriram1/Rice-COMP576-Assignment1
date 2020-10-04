__author__ = 'sai_sriram'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import three_layer_neural_network as tlnn

class DeepNeuralNetwork(tlnn.NeuralNetwork):
    def __init__(self, nn_input_dim, nn_hidden_dims, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dims: a list representing the number of hidden units in each of the n layers
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_n_layers = len(nn_hidden_dims)
        self.nn_hidden_dims = nn_hidden_dims
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.layers = [] # list of all Layers
        dims = [self.nn_input_dim] + self.nn_hidden_dims + [self.nn_output_dim]
        for i in range(self.nn_n_layers + 1):
            W_i = np.random.randn(dims[i], dims[i+1]) / np.sqrt(dims[i])
            b_i = np.zeros((1, dims[i+1]))
            if (i == self.nn_n_layers): # final layer gets softmax activation
                layer = Layer(W_i, b_i, 'softmax')
            else:
                layer = Layer(W_i, b_i, self.actFun_type)
            self.layers.append(layer)


    def feedforward(self, X):
        """
        Feedforward routine for the entire deep neural network.
        """
        for layer in self.layers:
            X = layer.feedforward(X)
        self.probs = X  # predictions are X after the last iteration.

    def backprop(self, X, y):
        """
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        """
        # convert labels to one-hot encoding (for example, if y[0] = 1, the first row of labels will be [0 1])
        labels = np.zeros((X.shape[0], 2))
        labels[:, 0] = y == 0
        labels[:, 1] = y == 1

        dZ_last = (self.probs - labels)
        last_layer = self.layers[-1]
        second_last_layer = self.layers[-2]
        dA = last_layer.backprop_last_layer(dZ_last, second_last_layer.a)
        for i in range(len(self.layers) - 2, -1, -1): # iterate backwards through the layers (not including the last or first layer)
            if (i == 0):
                break
            cur_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            dA = cur_layer.backprop(dA, prev_layer.a)
        
        # backprop very first layer separately since X (the input) is needed
        first_layer = self.layers[0]
        first_layer.backprop(dA, X)

    def calculate_loss(self, X, y):
        """
        Calculate the loss of the network after feedforward pass.
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        """
        self.feedforward(X)
      
        # convert labels to one-hot encoding (for example, if y[0] = 1, the first row of labels will be [0 1])
        labels = np.zeros((X.shape[0], 2))
        labels[:, 0] = y == 0
        labels[:, 1] = y == 1
        # Calculating the loss
        data_loss = np.sum(np.log(self.probs + 1e-8) * labels).astype(float)
        # Add regulatization term to loss (optional)
        reg_sum = 0
        for layer in self.layers: # compute the sum of squares of all weights and add them up
            reg_sum += np.sum(np.square(layer.weights))
        data_loss += self.reg_lambda / 2 * reg_sum
        return (-1/X.shape[0]) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(X, y)

            for layer in self.layers:
                # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
                layer.dW += self.reg_lambda * layer.weights
                # Gradient descent update for each layer
                layer.weights += -epsilon * layer.dW
                layer.biases += -epsilon * layer.db

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
    
    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        tlnn.plot_decision_boundary(lambda x: self.predict(x), X, y)

class Layer():
    """
    Class representing a single layer in the neural network. 
    Implements feedforward and backprop for a single layer of the network. 
    """
    def __init__(self, weights, biases, actFun_type):
        self.weights = weights
        self.biases = biases
        self.actFun_type = actFun_type

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''
        if (type == 'sigmoid'):
            return z * (1 - z) # element-wise multiply
        elif (type == "tanh"):
            return 1 - np.square(z) # element wise square
        elif (type == "relu"):
            z[z > 0] = 1
            z[z <= 0] = 0
            return z
        return None
    
    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''
        if (type == 'sigmoid'):
            return 1/(1 + np.exp(-z)) 
        elif (type == "tanh"):
            return np.tanh(z)
        elif (type == "relu"):
            return np.maximum(z, 0)
        elif (type == "softmax"):
            return np.exp(self.z) / np.sum(np.exp(self.z), axis=1, keepdims=True)
        return None

    def feedforward(self, X):
        """
        Compute feedforward pass of one layer of the neural network.
        """
        self.z = X.dot(self.weights) + self.biases # compute raw activations of hidden layer
        self.a = self.actFun(self.z, self.actFun_type) # apply activation function to hidden layer
        return self.a

    def backprop(self, da, X):
        """
        Compute the derivatives of this layer's weights and biases given da, the downstream gradient and
        X, the input to this layer in the feedforward pass.
        """
        dZ = self.diff_actFun(self.a, type=self.actFun_type) * da
        self.dW = X.T.dot(dZ)
        self.db = np.sum(dZ, axis=0)
        return dZ.dot(self.weights.T) # return downstream gradient for previous layer

    def backprop_last_layer(self, dZ, X):
        """
        Special method only for the final layer in the network, whose backprop routine looks a little different.
        """
        self.dW = X.T.dot(dZ)
        self.db = np.sum(dZ, axis=0)
        return dZ.dot(self.weights.T) # return downstream gradient for previous layer


def generate_circle_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(n_samples=200)
    return X, y
        

def main():
    # generate Make-Moons dataset and train neural net on it
    X, y = tlnn.generate_data()
    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dims=[3, 3] , nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

    # generate cicle dataset and train neural net on it
    X, y = generate_circle_data()
    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dims=[10, 10, 10] , nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)


if __name__ == "__main__":
    main()
