#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:17:33 2021

@author: erlend
"""

import numpy as np

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_categories,
            output_func,
            hidden_layer_sizes=[50], #list of number of neurons in each layer, last is output
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
        self.hidden_layer_sizes = hidden_layer_sizes #nodes in layers

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        
        #Neurons in input layer
        self.weights = [np.random.randn(self.n_features, self.hidden_layer_sizes[0])] 
        self.bias = [np.zeros(self.hidden_layer_sizes[0]) + 0.01]
        
        #Neurons in hidden layers
        for new_neurons in self.hidden_layer_sizes[1:]:
            self.weights.append(np.random.randn(self.weights[-1].shape[1], new_neurons))
            self.bias.append(np.zeros(new_neurons) + 0.01)
       
        #output layer
        self.weights.append(np.random.randn(self.weights[-1].shape[1], self.n_categories))
        self.bias.append(np.zeros(self.n_categories) + 0.01)
        
        #scale weights so they match x-input..
        scale = 1/max(self.X_data_full.ravel())
        self.weights = [w * scale for w in self.weights]

        self.a_h = [i for i in range(len(self.hidden_layer_sizes) + 1)] #+1 is outputlayer
        #TODO
        #Implement general f', derivative in backpropagation
        #List of activation functions

    def sigmoid(self, x):
        #return np.tanh(x)
        return 1/(1 + np.exp(-x))

    def feed_forward(self):
    
        # feed-forward for training
         
        self.a_h[0] = self.sigmoid(np.matmul(self.X_data, self.weights[0]) + self.bias[0])
        for l in range(1, len(self.hidden_layer_sizes)):
            self.a_h[l] = self.sigmoid(np.matmul(self.a_h[l - 1], self.weights[l])+self.bias[l])
        
        self.a_h[-1] = self.output_func(np.matmul(self.a_h[-2], self.weights[-1]) + self.bias[-1])

    def backpropagation(self):
        error = self.a_h[-1] - self.Y_data #delta output
        for l in range(len(self.hidden_layer_sizes), 0, -1):
            weights_gradient = np.matmul(self.a_h[l - 1].T, error)
            bias_gradient = np.sum(error, axis=0) #kan ve fler kolonner

            if self.lmbd > 0.0:
                weights_gradient += self.lmbd * self.weights[l]

            self.weights[l] -= self.eta * weights_gradient
            self.bias[l] -= self.eta * bias_gradient

            error = np.matmul(error, self.weights[l].T)*self.a_h[l-1]*(1 - self.a_h[l-1])

        weights_gradient = np.matmul(self.X_data.T, error)
        bias_gradient = np.sum(error, axis=0)

        if self.lmbd > 0.0:
            weights_gradient += self.lmbd * self.weights[0]

        self.weights[0] -= self.eta * weights_gradient
        self.bias[0] -= self.eta * bias_gradient


    def feed_forward_out(self, X):
        self.a_h[0] = self.sigmoid(np.matmul(X, self.weights[0]) + self.bias[0])
        for l in range(1, len(self.hidden_layer_sizes)):
            self.a_h[l] = self.sigmoid(np.matmul(self.a_h[l - 1], self.weights[l])+self.bias[l])
        
        self.a_h[-1] = self.output_func(np.matmul(self.a_h[-2], self.weights[-1]) + self.bias[-1])
        return self.a_h[-1]


    def predict(self, X):
        return self.feed_forward_out(X)

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
            print("Epoch: ", i)
