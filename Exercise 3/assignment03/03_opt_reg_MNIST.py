"""
USAGE : To run Nesterov with L2, run 'python 03_opt_reg_MNIST.py -- opt nesterov --reg L2'
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--opt', default='sgd', help='sgd or nesterov [default: sgd]')
parser.add_argument('--reg', default= None, help='L1/L2/None [default: None]')
FLAGS = parser.parse_args()


# YOUR CODE HERE : Specify the number of hidden units for each layer

n_input 	= 784 # e.g. MNIST data input (img shape: 28*28)
n_hidden1	= # hidden layer num units 
n_hidden2	= # hidden layer num units
n_classes  	= 10 # e.g. MNIST total classes (0-9 digits)
mnist 		= input_data.read_data_sets('MNIST_data', one_hot=True)

def mlp_config(n_input,n_hidden1,n_hidden2,n_classes):
# tf Graph variables
	x = tf.placeholder("float", [None, n_input], name='x')
	y = tf.placeholder("float", [None, n_classes], name='y')

	# Store layers weight & bias
	stddev = 0.1 # <== This greatly affects accuracy!! 
	weights = {
	    'h1':  tf.Variable(tf.random_normal([n_input, n_hidden1], stddev=stddev)),
	    'h2':  tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=stddev)),
	    'out': tf.Variable(tf.random_normal([n_hidden2, n_classes], stddev=stddev))
	}
	biases = {
	    'b1':  tf.Variable(tf.random_normal([n_hidden1])),
	    'b2':  tf.Variable(tf.random_normal([n_hidden2])),
	    'out': tf.Variable(tf.random_normal([n_classes]))
	}
	return x,y,weights,biases

def mlp_model(x,y,weights,biases):

	# YOUR CODE HERE : Create your neural network based on the specifications given in the 
	# 	           exercise sheet (number of hidden units, drop-out rate etc). 
	#		   The function should return pred and logits as two tensors. 	
        
	return pred, logits

def compute_loss(logits,weights,y,regularizer):
	#define parameters
	weight_decay_l1 = 1e-4
	weight_decay_l2 = 1e-5
	# Define loss
	with tf.name_scope('Loss'):

		cost = # Compute cross-entropy loss

		# Write code for doing L2/L1/no regularization on the network parameters.
		if (regularize == None): # no regularization
			loss = tf.reduce_mean(cost)
		elif (regularizer == "L2"): 
			loss = # code for L2 regularization
		elif (regularizer == "L1"): 
		        loss = # code for L1 regularization

	tf.summary.scalar("loss",loss)
        return loss
	
def cal_accuracy(pred,y):
	with tf.name_scope('accuracy'):
	    with tf.name_scope('correct_prediction'):
	      correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	    with tf.name_scope('accuracy'):
	      accuracy 		 = # Compute accuracy
	tf.summary.scalar('accuracy', accuracy)
        return accuracy

def train_network(optimizer='sgd',regularizer=None):
	

	# config and build model
	x,y,weights,biases = mlp_config(n_input,n_hidden1,n_hidden2,n_classes)
	pred, logits 	   = mlp_model(x,y,weights,biases) # Check comments inside 'mlp_model' function
  
        loss 	 = compute_loss(logits,weights,y,regularizer)
        accuracy = cal_accuracy(pred,y)
	
	# You can try out different learning rates but it should be the same for both 
	# the optimizers for proper comparison.

	if (optimizer == 'sgd'): 
		learning_rate = 0.01 
		train_step    = # add code for SGD optimizer

	elif (optimizer == 'nesterov'):
                momentum      = 0.9
		learning_rate = 0.01
		train_step    = # add code for Nesterov optimizer
	
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	# Merge all the summaries and write them out to corresponding directories
	
	merged 	     = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter( './train', sess.graph)
	test_writer  = tf.summary.FileWriter('./test')
	tf.reset_default_graph() 

	# Report your results for 10000 iterations. If time does not allow, you can also report
	# results for 5000 iterations.

	for i in range(10000):
	    batch = mnist.train.next_batch(1000) # Fetch batch

	    if (i % 100) == 0:  # Record summaries ( loss etc) and test-set accuracy
	      summary, acc = # Run session to compute summary and accuracy
	      test_writer.add_summary(summary, i)
	      print('Test Accuracy at step %s: %s' % (i, acc))
	    else:
	      summary,_ = sess.run([merged,train_step], feed_dict={x: batch[0], y: batch[1]})
	      train_writer.add_summary(summary, i)

	train_writer.close()
	test_writer.close()
	
	print ("Accuracy using tensor flow is:")
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
	writer = tf.summary.FileWriter('mlp_model',sess.graph)
	writer.add_graph(sess.graph)

if __name__=="__main__":
	optimizer   = FLAGS.opt;
	regularizer = FLAGS.reg;
	print('Optimizer : %s Regularizer : %s'%(optimizer,regularizer))
	train_network(optimizer,regularizer)
	
