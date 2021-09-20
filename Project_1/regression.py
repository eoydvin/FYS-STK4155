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
    
    def var_beta(self):
        return 
    
    
    
    
    
    
    
    