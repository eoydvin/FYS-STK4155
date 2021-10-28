#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:01:26 2021

@author: erlend
"""

import numpy as np

class OLS(object):
    """
    Fits a polynomial to given data using OLS. Uses stcastic gradient descent.    

    """
    def __init__(self, p, learning_rate, alpha, batch_size=32, 
                 max_iter=100, n_epochs=50):
        """        
        Fit polynomial using SGD. 
        
        Parameters
        ----------
        p : order of polynomial
        learning_rate : learning rate
        batch_size : batch_size of SDG
        max_iter : maximum number of iterations SGD
        
        """
        
        self.p = p
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
    
    def p_design(self, X):
        """
        Construct design matrix for a polynomial fit. 

        """
        M = np.zeros([X.shape[0], int((self.p+1)*(self.p+2)/2)])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1-j): # x **i
                M[:, c] = (X[:, 0]**i)*(X[:, 1]**j)
                c += 1
        return M
            
        
    def fit(self, X, z):
        """
        Fit polynomial using SGD. 
        
        Parameters
        ----------
        X : Matrix of features and samples
        z : Array of corresponding realizations

        """
        M = self.p_design(X)
        m = int(M.shape[0]/self.batch_size) # number minibatches       
        
        w = np.zeros(M.shape[1]).reshape(-1, 1) 
        v = np.zeros(M.shape[1]).reshape(-1, 1) 
        # maybe use random numbers that us prop. to eigenvalues? 

        for epoch in range(self.n_epochs):
            # lag batcher her:) 
            Xz = np.hstack([M, z])
            np.random.shuffle(Xz) #shuffle
            Xz_split = np.array_split(Xz, m) #split into batches
            
            for batch in Xz_split:
                g = (1.0/m) *batch[:, 0:-1].T @ ((
                    batch[:, 0:-1] @ w) - batch[:, -1].reshape(-1, 1))
                v = self.alpha*v - self.learning_rate*g
                w = w + v

        self.w = w
    
    def predict(self, X):
        """
        Predict f(x, y) = z for given X[x, y] a
        
        Parameters
        ----------
        X : Matrix of features and samples
        
        :return: array of predicted z values
        """
        M = self.p_design(X)
        return M @ self.w
    
    def MSE(self, X, z):
        """
        Calculate MSE using 
        
        MSE = 1/n * sum_i ((z_pred_i - z_i)**2)

        :param x: array of x coordinates
        :param y: array of y coordinates
        :param z: array of known z values      

        """
        M = self.p_design(X)    
        z_pred = M @ self.w   
        
        #print(self.w)
        #print(z_pred.shape)
        return np.mean((z_pred - z.reshape([M.shape[0], 1]))**2) 
        
    def R2(self, X, z):
        """
        Calculate R2 using 
        
        R2 = 1 - (sum_i ((z_pred_i - z_i)**2)) / (sum_i ((z_i_mean - z_i)**2))

        :param x: array of x coordinates
        :param y: array of y coordinates
        :param z: array of known z values      

        """
        M = self.p_design(X)    
        z_pred = M @ self.w   
        
        return 1 - np.sum((z_pred - z.reshape([M.shape[0], 1]))**2)/np.sum(
            (np.mean(z) - z.reshape([M.shape[0], 1]))**2)    
    
