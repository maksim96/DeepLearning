# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:08:05 2018

@author: Frank
"""

from load_mnist import * 
import hw1_knn  as mlBasics  
import numpy as np 
from sklearn.metrics import confusion_matrix
   
  
# Load data - two class 
##X_train, y_train = load_mnist('training' , [0,1] )
##X_test, y_test = load_mnist('testing'  ,  [0,1]  )

# Load data - ALL CLASSES
X_train, y_train = load_mnist('training'  )
X_test, y_test = load_mnist('testing'   )


samind=np.zeros(0)      # collect all the sample numbers
sample_labels=np.zeros(0)       # saving the labels

for i in range(0,10):   # get the 100 training samples from the 10 classes

    val= np.isin(y_train,i)                             # filter out the indices, where label is i, gives an array with boolean values false and true
    ind=np.array(np.where(val))                         # filters the corresponding indices
    ind= np.reshape(ind,(ind.shape[1]))                 # make it 1 dimensional
    s=np.random.choice(ind,100 ,replace=False )         # randomply sample 100 indices

    sample_labels=np.concatenate((sample_labels,np.full((100),i)),axis=0)   # save the corresponing labels for easier use later
    samind=np.concatenate((samind,s),axis=0)            # concatenate new sample results to existing ones
samind=samind.astype(int)
samples=X_train[samind]                     # the 1000 training samples

  

# Reshape the image data into rows  
samples = np.reshape(samples, (samples.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

  
# Test on test data   
# Compute distances:
dists =  mlBasics.compute_euclidean_distances(samples,X_test) 

# Calculate the test predictions for k=1 and k=5: 
y_test_pred_k1 = mlBasics.predict_labels2(dists, y_train,1 ) # prediction for k=1

y_test_pred_k5 = mlBasics.predict_labels2(dists, y_train,5 ) # prediction for k=5

#3) Report results
# you should get following message '99.91 of test examples classified correctly.'
print ("Accuracy for k=1: ", '{0:0.02f}'.format(  np.mean(y_test_pred_k1==y_test)*100), "of test examples classified correctly.")
print ("Accuracy for k=5: ",'{0:0.02f}'.format(  np.mean(y_test_pred_k5==y_test)*100), "of test examples classified correctly.")

cmat_k1=confusion_matrix(y_test,y_test_pred_k1)             # calculate the confusion matrix
cmat_k5=confusion_matrix(y_test,y_test_pred_k5)
print("Confusion matrix for k=1: ")
print(cmat_k1)
print("Confusion matrix for k=5: ")
print(cmat_k5)

