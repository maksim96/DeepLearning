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
n_hidden1	= 800
n_hidden2	= 800
n_classes  	= 10 # e.g. MNIST total classes (0-9 digits)
mnist 		= input_data.read_data_sets('MNIST_data', one_hot=True)

def mlp_config(n_input,n_hidden1,n_hidden2,n_classes):
# tf Graph variables
	x = tf.placeholder("float", [None, n_input], name='x')
	y = tf.placeholder("float", [None, n_classes], name='y')

	# Store layers weight & bias
	stddev = 1 # <== This greatly affects accuracy!! 
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

	dropped_input = tf.nn.dropout(x,keep_prob=0.8)

	layer_1 = tf.nn.relu(tf.add(tf.matmul(dropped_input, weights['h1']), biases['b1']))    
	dropped_hidden_1 = tf.nn.dropout(layer_1,keep_prob=0.5)

	layer_2 = tf.nn.relu(tf.add(tf.matmul(dropped_hidden_1, weights['h2']), biases['b2']))
	dropped_hidden_2 = tf.nn.dropout(layer_2,keep_prob=0.5)

    # Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(dropped_hidden_2, weights['out']) + biases['out']
	return tf.nn.softmax(out_layer), out_layer

def compute_loss(logits,weights,y,regularizer):
	#define parameters
	weight_decay_l1 = 1e-4
	weight_decay_l2 = 1e-5
	# Define loss
	with tf.name_scope('Loss'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
		if (regularizer == None):
			loss = cost
		elif (regularizer == "L2"): 
			loss = cost + weight_decay_l2*(tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out']))
		elif (regularizer == "L1"): 
			loss = cost + weight_decay_l1*(tf.math.abs(weights['h1']) + tf.math.abs(weights['h2']) + tf.math.abs(weights['out']))
		tf.summary.scalar("loss",loss)
		return loss
	
def cal_accuracy(pred,y):
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.metrics.accuracy(labels=y, predictions=pred)[1]
	tf.summary.scalar('accuracy', accuracy)
	return accuracy

def train_network(optimizer='sgd',regularizer=None):
	# config and build model
	x,y,weights,biases = mlp_config(n_input,n_hidden1,n_hidden2,n_classes)
	pred, logits 	   = mlp_model(x,y,weights,biases) # Check comments inside 'mlp_model' function
  
	loss = compute_loss(logits,weights,y,regularizer)
	accuracy = cal_accuracy(pred,y)
	
	# You can try out different learning rates but it should be the same for both 
	# the optimizers for proper comparison.

	if (optimizer == 'sgd'): 
		learning_rate = 0.001 
		train_step    = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	elif (optimizer == 'nesterov'):
		momentum      = 0.9
		learning_rate = 0.001
		train_step    = tf.train.MomentumOptimizer(learning_rate,momentum,use_nesterov=True).minimize(loss)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	sess.run(tf.local_variables_initializer())
	
	# Merge all the summaries and write them out to corresponding directories
	
	merged 	     = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter( './train', sess.graph)
	test_writer  = tf.summary.FileWriter('./test')
	tf.reset_default_graph() 

	# Report your results for 10000 iterations. If time does not allow, you can also report
	# results for 5000 iterations.

	for i in range(1000):
		batch = mnist.train.next_batch(1000) # Fetch batch

		if (i % 100) == 0:  # Record summaries ( loss etc) and test-set accuracy
			summary, acc = sess.run([merged,accuracy], feed_dict={x: batch[0], y: batch[1]})
			test_writer.add_summary(summary, i)
			print('Test Accuracy at step %s: %s' % (i, acc))
		else:
			summary,_,acc = sess.run([merged,train_step, accuracy], feed_dict={x: batch[0], y: batch[1]})
			train_writer.add_summary(summary, i)
			#print('Training	 Accuracy at step %s: %s' % (i, acc))

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
	
