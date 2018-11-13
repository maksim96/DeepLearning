# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:08:05 2018

@author: Frank & Andreas
"""

from load_mnist import *
import hw1_knn as mlBasics
import numpy as np
from sklearn.metrics import confusion_matrix
from pprint import pprint


def pred_accuracy(train_X, train_y, test_X, test_y, k):
    # compute distances
    dists = mlBasics.compute_euclidean_distances(train_X, test_X)

    # calculate the predictions
    pred = mlBasics.predict_labels2(dists, train_y, k)

    # return accuracy
    return pred, np.mean(pred == test_y) * 100


def five_fold_cv(samples_X, samples_y, k):
    # 5 fold cross validation
    # get shape zero divided by 5 for folds (assumes number of rows is divisible by 5)
    shape_zero = int(samples_X.shape[0] / 5)
    # create bool array to use as mask later
    bool_arr = np.full((samples_X.shape[0]), True, dtype=bool)
    tru = np.full(shape_zero, True, dtype=bool)
    fal = np.full(shape_zero, False, dtype=bool)

    # list for mean computation
    acc_list = []

    for i in range(5):
        # create the train data for current round
        # takes one fifth of samples out of the set for training
        bool_arr[i * shape_zero: shape_zero + i * shape_zero] = fal
        train_X = samples_X[bool_arr][:].copy()
        train_y = samples_y[bool_arr][:].copy()

        # invert bool_arr for testing selection
        bool_arr.__invert__()
        # takes the remaining 4/5 of samples for testing
        test_X = samples_X[bool_arr].copy()
        test_y = samples_y[bool_arr].copy()

        # calculate prediction/accuracy tuple
        pred_acc = pred_accuracy(train_X, train_y, test_X, test_y, k)

        # concat accuracy to acc_list
        acc_list.append(pred_acc[1])

        # make bool_arr full true array again.
        bool_arr.__invert__()
        bool_arr[i * shape_zero: shape_zero + i * shape_zero] = tru

    # return average accuracy
    return np.average(acc_list, axis=0)


def main():
    # Load data - ALL CLASSES
    X_train, y_train = load_mnist('training')
    X_test, y_test = load_mnist('testing')

    # collect all the sample indices
    samind = np.zeros(0)

    for i in range(0, 10):  # get the 100 training samples from the 10 classes
        # filter out the indices, where label is i, gives an array with boolean values false and true
        val = np.isin(y_train, i)
        # filters the corresponding indices
        ind = np.array(np.where(val))
        # make it 1 dimensional
        ind = np.reshape(ind, (ind.shape[1]))
        # randomly sample 100 indices
        s = np.random.choice(ind, 100, replace=False)
        # concatenate new sample results to existing ones
        samind = np.concatenate((samind, s), axis=0)

    samind = samind.astype(int)

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # get the 1000 samples and corresponding labels
    samples_X = X_train[samind]
    samples_y = y_train[samind]

    # calculate predictions/accuracy tuples for k=1 and k=5
    pred_acc_one = pred_accuracy(samples_X, samples_y, X_test, y_test, 1)
    pred_acc_five = pred_accuracy(samples_X, samples_y, X_test, y_test, 5)

    # Report results
    print("Accuracy for k=1: ", '{0:0.02f}'.format(pred_acc_one[1]),
          "of test examples classified correctly.")
    print("Accuracy for k=5: ", '{0:0.02f}'.format(pred_acc_five[1]),
          "of test examples classified correctly.")

    # calculate the confusion matrix
    cmat_k1 = confusion_matrix(y_test, pred_acc_one[0])
    cmat_k5 = confusion_matrix(y_test, pred_acc_five[0])
    print("Confusion matrix for k=1: ")
    print(cmat_k1)
    print("Confusion matrix for k=5: ")
    print(cmat_k5)

    # c)
    # Dict to receive the averages for the different k's
    avg_dict = {}

    # Do five fold cross validation
    for k in range(1, 16):
        # calculate the average accuracy of knn using five fold cross validation for current k
        avg_dict[k] = five_fold_cv(samples_X, samples_y, k)

    print("Accuracies for k from 1 to 15 using 5 fold CV:")
    pprint(avg_dict)
    print("As can be seen k=1 performs the best with an average accuracy of 100%.")

    # d)
    # calculate predictions/accuracy tuple for k=1 for all 60,000 images.
    # useless to run this, since now python wants to use more than 10GB of RAM, while my PC only has 8...
    # therefore too much slowdown from all the swapping...
    #pred_acc_one_all = pred_accuracy(X_train, y_train, X_test, y_test, 1)
    #print("Accuracy for complete set:", pred_acc_one_all[1])


if __name__ == '__main__':
    main()
