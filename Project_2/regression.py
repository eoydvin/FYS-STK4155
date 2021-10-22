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
    def __init__(self, p, learning_rate, batch_size=256, max_iter=100):
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        
    def fit(x, z):
        """
        Fit polynomial using SGD. 
        
        Parameters
        ----------
        x : Matrix of features and samples
        z : Array of corresponding realizations
  

        Returns
        -------
        None.

        """
        # Construct design matrix
        n = len(x)
        M = np.zeros([n, int((p+1)*(p+2)/2)])
        c = 0
        for j in range(0, p+1): # y**j
            for i in range(0, p+1-j): # x **i
                M[:, c] = (x**i)*(y**j)
                c += 1
        # SGD:
            #del inn i b batcher,
            #beregn b for en batch
            #send beta videre til neste batch

        self.beta_OLS = np.linalg.pinv(M.T @ M) @ M.T @ z.reshape([n, 1])
        
        #np.linalg.pinv(M) @ z.reshape([n, 1])
        self.p = p
        self.design = design
    
    def R2(self, x, y, z):
        """
        Calculate R2 using 
        
        R2 = 1 - (sum_i ((z_pred_i - z_i)**2)) / (sum_i ((z_i_mean - z_i)**2))

        :param x: array of x coordinates
        :param y: array of y coordinates
        :param z: array of known z values      

        """
        # Construct design matrix
        n = len(x)
        M = np.zeros([n, int((self.p+1)*(self.p+2)/2)])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1 - j): # x **i
                M[:, c] = (x**i)*(y**j)
                c += 1
        
        z_pred = M @ self.beta_OLS    
        return 1 - np.sum((z_pred - z.reshape([n, 1]))**2)/np.sum(
            (np.mean(z) - z.reshape([n, 1]))**2)    
                
    def predict(self, x, y):
        """
        Predict f(x, y) = z for given x and y
        
        :param x: array of x coordinates
        :param y: array of y coordinates
        
        :return: array of predicted z values
        """
        # Construct design matrix
        n = len(x)
        M = np.zeros([n, int((self.p+1)*(self.p+2)/2)])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1 - j): # x **i
                M[:, c] = (x**i)*(y**j)
                c += 1
        return M @ self.beta_OLS
    
    def MSE(self, x, y, z):
        """
        Calculate MSE using 
        
        MSE = 1/n * sum_i ((z_pred_i - z_i)**2)

        :param x: array of x coordinates
        :param y: array of y coordinates
        :param z: array of known z values      

        """
        # Construct design matrix
        n = len(x)
        M = np.zeros([n, int((self.p+1)*(self.p+2)/2)])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1 - j): # x **i
                M[:, c] = (x**i)*(y**j)
                c += 1
       
        z_pred = M @ self.beta_OLS    
        
        return np.mean((z_pred - z.reshape([n, 1]))**2) 
        
    
    def var_beta(self, x, y, z):
        """
        Calculate variance in beta by comparing predicted values with test 
        values according to Hastie et al:
        
        var(beta) = inv(X.T X) * var(y), where:    
        var(y) = 1 / (N - p - 1) * sum((z_i  - z_pred_i)**2)
        
        :param x: array of x coordinates
        :param y: array of y coordinates
        :param z: array of known z values
        
        :return: array of estiamted variance for beta values
        """
        # Construct design matrix
        n = len(x)
        M = np.zeros([n, int((self.p+1)*(self.p+2)/2)])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1 - j): # x **i
                M[:, c] = (x**i)*(y**j)
                c += 1

        
        z_pred = M @ self.beta_OLS
        variance_z = (1/(n - self.p - 1))*np.sum((
            z_pred - z.reshape([n, 1]))**2)
        # Test ved bruk av M fra init gir samme resultat
        return np.diagonal(variance_z*np.linalg.pinv(M.T @ M))  

