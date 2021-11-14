#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:17:33 2021

@author: erlend
"""

import autograd.numpy as np
from autograd import elementwise_grad as egrad 
import sys


class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_categories,
            output_func,
            activation_func = 'logistic',
            hidden_layer_sizes=[50, 10],
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.01):

        self.output_func = output_func
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_categories = n_categories
        self.hidden_layer_sizes = hidden_layer_sizes 

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        
        #Initialize weigths and biases
        # - w: normal distribution
        # - b: sligthly more than zero
        #Input layer
        self.weights = [np.random.randn(
            self.n_features, self.hidden_layer_sizes[0])] 
        self.bias = [np.zeros(self.hidden_layer_sizes[0]) + 0.01]
        
        #Hidden layers
        for new_neurons in self.hidden_layer_sizes[1:]:
            self.weights.append(
                np.random.randn(self.weights[-1].shape[1], new_neurons))
            self.bias.append(np.zeros(new_neurons) + 0.01)
        
        #output layer
        self.weights.append(np.zeros(
            [self.weights[-1].shape[1], self.n_categories]))
        self.bias.append(np.zeros(self.n_categories) + 0.01)
        
        #scale weights so they match x-input..
        scale = 1/max(self.X_data_full.ravel())
        self.weights = [w * scale for w in self.weights]

        self.a_h = [i for i in range(len(self.hidden_layer_sizes) + 1)] #+1 is outputlayer
        #TODO
        # Implement general f', derivative in backpropagation
        # List of activation functions
        
        if activation_func == 'logistic':
            from neural_network import logistic
            from neural_network import logistic_derivative
            self.activation_func = logistic
            self.d_activation_func = logistic_derivative
        
        elif activation_func == 'relu':
            from neural_network import relu
            from neural_network import relu_derivative

            self.activation_func = relu
            self.d_activation_func = relu_derivative

        elif activation_func == 'leakyrelu':
            from neural_network import leakyrelu
            self.activation_func = leakyrelu
            self.d_activation_func = egrad(self.activation_func)
    
    def _logistic(self, x):
        return 1 / (1 + np.exp(-x)) 

            
    def feed_forward(self):
    
        # feed-forward for training
        self.a_h[0] = self.activation_func( 
            np.matmul(self.X_data, self.weights[0]) + self.bias[0])
        for l in range(1, len(self.hidden_layer_sizes)):
            self.a_h[l] = self.activation_func( 
                np.matmul(self.a_h[l - 1], self.weights[l])+self.bias[l])
        
        self.a_h[-1] = self.output_func(
            np.matmul(self.a_h[-2], self.weights[-1]) + self.bias[-1])

    def backpropagation(self):
        error = self.a_h[-1] - self.Y_data #delta output
       
        for l in range(len(self.hidden_layer_sizes), 0, -1):
            weights_gradient = np.matmul(self.a_h[l - 1].T, error)
            bias_gradient = np.sum(error, axis=0) 


            if self.lmbd > 0.0:
                weights_gradient += self.lmbd * self.weights[l]

            self.weights[l] -= self.eta * weights_gradient
            self.bias[l] -= self.eta * bias_gradient
            
            
            #error = np.matmul(
            #    error, self.weights[l].T) *self.a_h[l-1]*(1 - self.a_h[l-1])
            
            error = np.matmul(
                error, self.weights[l].T)* self.d_activation_func( 
                        np.matmul(self.a_h[l - 1], self.weights[l]) + self.bias[l]) 

        weights_gradient = np.matmul(self.X_data.T, error)
        bias_gradient = np.sum(error, axis=0)

        if self.lmbd > 0.0:
            weights_gradient += self.lmbd * self.weights[0]

        
        self.weights[0] -= self.eta * weights_gradient
        self.bias[0] -= self.eta * bias_gradient


    def feed_forward_out(self, X):
        self.a_h[0] = self.activation_func(np.matmul( 
            X, self.weights[0]) + self.bias[0])
        for l in range(1, len(self.hidden_layer_sizes)):
            self.a_h[l] = self.activation_func(  
                np.matmul(self.a_h[l - 1], self.weights[l])+self.bias[l])
        
        self.a_h[-1] = self.output_func(
            np.matmul(self.a_h[-2], self.weights[-1]) + self.bias[-1])
        return self.a_h[-1]


    def predict(self, X):
        return self.feed_forward_out(X)

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with no replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                self.feed_forward()
                self.backpropagation()


def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1.0 - logistic(x))

def relu(x):
    np.clip(x, a_min = 0.0, a_max = sys.float_info.max, out = x)
    return x

def relu_derivative(self, x) :
    return relu(np.heaviside(x, 0.0) )

def leakyrelu(self, x) :
    return (x >= 0.0) * x + (x < 0.0) * (self.alpha * x)
