import numpy as np
import random
from scipy.special import expit, softmax
import sys



class Sigmoid:
    def classic(x):
        return expit(x)
    
    def derivative(x):
        return Sigmoid.classic(x)*(1-Sigmoid.classic(x))

class Relu:
    def classic(x):
        return x * (x > 0)

    def derivative(x):
        return 1 * (x > 0)
     
class Linear:
    def classic(x):
        return x
    
    def derivative(x):
        return 1 * (x==x)
    
class Softmax:
    def classic(x):
        rez = np.array([softmax(xi) for xi in x.T]).T
        return rez
    
    def derivative(x):
        None



class Cost:
    def binary_loss(y, y_hat):
        return -(1-y)/(1-y_hat)+(y/y_hat)
    
    def binary_crossentropy(y, y_hat):
        return y*np.log(y_hat)+(1-y)*np.log(1-y_hat)
    
    def crossentropy(y, y_hat):
        return (y*np.log(y_hat)).sum()
    
    def cost(func, y, y_hat):
        return -func(y, y_hat).sum()


class Weight_Init:
    def random_init(layer_size, input_size):
        w = np.random.randn(layer_size, input_size)*0.01
        b = np.zeros(shape=(layer_size, 1))
        return w, b


class Dense:
    def __init__(self, layer_size, activation):
        self.layer_size = layer_size
        self.activation = activation
        
    def activate(self, input_dim, learning_rate):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.W, self.B = Weight_Init.random_init(self.layer_size, input_dim)
    
    def forward(self, X):
        self.X = X
        self.Z = np.dot(self.W, X)+self.B
        self.A = self.activation.classic(self.Z)
        return self.A
    
    def backwards(self, dA_last, m_examples):
        if self.activation == Softmax:
            dZ = dA_last
        else:
            dZ = dA_last * self.activation.derivative(self.Z)
        dW = np.dot(dZ, self.X.T)/m_examples
        dB = np.expand_dims(1/m_examples * np.sum(dZ, axis=1), axis=1)
        self.update_weights(dW, dB)
        dA_last = np.dot(self.W.T, dZ)
        return dA_last
    
    def update_weights(self, dW, dB):
        self.W -= self.learning_rate * dW
        self.B -= self.learning_rate * dB


class Model:
    def __init__(self, input_size, batch_size, learning_rate = 0.1):
        self.layers = []
        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def add(self, layer):
        if self.layers == []:
            input_dim = self.input_size
        else:
            input_dim = self.layers[-1].layer_size
        layer.activate(input_dim, self.learning_rate)
        self.layers += [layer]
        
    def predict(self, X):
        last_activation = X
        for layer in self.layers:
            last_activation = layer.forward(last_activation)
        return last_activation
    
    def backprop(self, last_activation, Y):
        dA = last_activation - Y
        for layer in self.layers[::-1]:
            dA = layer.backwards(dA, self.batch_size)
            
    def calculate_cost(self, y, y_hat):
        self.cost = Cost.cost(Cost.crossentropy, y, y_hat)
        
    def print_log(self, iteration):
        sys.stdout.flush()
        sys.stdout.write('Iteration: ' + str(iteration+1) + ' - Cost: ' + str(self.cost)+ '\r')
            
    def train(self, X, Y, num_iterations):
        for i in range(num_iterations):
            last_activation = self.predict(X)
            self.calculate_cost(Y, last_activation)
            self.print_log(i)
            self.backprop(last_activation, Y)


