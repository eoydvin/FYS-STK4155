#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:01:26 2021

@author: erlend
"""

import numpy as np

class OLS(object):
    """
    Fits a polynomial to given data using OLS.         
    """
    def __init__(self, x, y, z, p):
        """
        Sets up design matrix for a k-order polynomial. 
        
        :param x: x value for given z=f(x,y)
        :param y: y value for given z=f(x,y)
        :param z: z value for given z=f(x,y) 
        :param p: order p of polynomial
        """
        
        # Construct design matrix
        n = len(x)
        M = np.zeros([n, (p+1)**2])
        c = 0
        for j in range(0, p+1): # y**j
            for i in range(0, p+1): # x **i
                M[:, c] = (x**i)*(y**j)
                c += 1
        
        self.beta_OLS = np.linalg.pinv(M) @ z.reshape([n, 1])
        self.p = p
    
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
        M = np.zeros([n, (self.p+1)**2])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1): # x **i
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
        M = np.zeros([n, (self.p+1)**2])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1): # x **i
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
        M = np.zeros([n, (self.p+1)**2])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1): # x **i
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
        M = np.zeros([n, (self.p+1)**2])
        c = 0
        for j in range(0, self.p+1): # y**j
            for i in range(0, self.p+1): # x **i
                M[:, c] = (x**i)*(y**j)
                c += 1
        
        z_pred = M @ self.beta_OLS
        variance_z = (1/(n - self.p - 1))*np.sum((
            z_pred - z.reshape([n, 1]))**2)
        # Test ved bruk av M fra init gir samme resultat
        return np.diagonal(variance_z*np.linalg.pinv(M.T @ M))  
    