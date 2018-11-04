# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np 
 

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    return sigmoid(np.dot(X,W) + b)
    
 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return 1/(1 + np.exp(-a))
    

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
    """
    s = predict(X, W, b)
    diff = y - s
    loss = np.dot(diff,diff)
    grad_w = -2*np.sum(X.T*s*(1-s)*diff, axis=1)
    grad_b = -2*np.sum(s*(1-s)*diff)

    return loss, grad_w, grad_b

def train(X,y,W,b, num_iters=1000, eta=0.001):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b
    Returns also all loss values during the optimization
    """

    all_losses = []

    for i in range(num_iters):

        loss,grad_w, grad_b = l2loss(X,y,W,b)
        W -= eta*grad_w
        b -= eta*grad_b
        all_losses.append(loss)
    return W,b,all_losses
