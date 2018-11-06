# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 
 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array 
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """  
    dist = -2 * np.dot(X, Y.transpose()) + np.sum(Y**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis] # use idea: (x-y)^2 = x^2 +y^2 -2*xy for faster computation
    return dist

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    indmin = np.argmin(dists,axis=0)  # get the index of the entry with minimum value for each testing example (= nearest training point)
    return labels[indmin]   # return the corresponding labels as prediction
    


def predict_labels2( dists, labels, k):                         # this function is needed for the second part in the main file
    dist_local=np.copy(dists)                                   # copy the distance array, because some values will be changed in the process
    
    all_labels = np.zeros((0,0))                                 # save all labels for later majority voting
    for j in range(0,k):
        indmin = np.argmin(dist_local,axis=0)                   # get the index of the entry with minimum value for each testing example (= nearest training point)

        for i in range(0, dist_local.shape[1]):                 # substitute the minimal values with a high value (inf) to find the next minimum (required for k min ) 
            dist_local[indmin[i]][i]=np.inf
        t_label=labels[indmin]
        t_label=np.reshape(t_label,(t_label.shape[0],1))        # reshape the labels for the concatenatinon

        if(j==0):                                               # only in first iteration save labels directly
            all_labels=np.copy(t_label)
        else:                                                   # later concatenate the labels
            all_labels=np.concatenate((all_labels,t_label),axis=1)


    all_labels=all_labels.astype(int)   # convert to int 
    
    labels=np.zeros(all_labels.shape[0]) # save here the labels after the majority voting

    for i in range(0,all_labels.shape[0]):   # go through all rows
        #if(i==0):   # testing
        #    print(all_labels[i])
        a=np.bincount(all_labels[i])    # do a bincount 
        labels[i]=np.argmax(a)          # choose the label with most appearances
        #if(i==0):        
        #    print(labels [i])
          
    return labels   # return the corresponding labels as prediction
    