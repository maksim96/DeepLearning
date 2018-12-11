from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("model_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 300
display_step = 1

# Network Parameters
""" Changed hidden layer size from 256 to 800.
"""
n_hidden_1 = 800  # 1st layer number of neurons
n_hidden_2 = 800  # 2nd layer number of neurons
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
""" Already uses relu for activation.
    What has to be implemented are:
        dropout layers:
            20% at input
            50% per hidden layer
"""


def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

""" Here the code from the git, using softmax and cross entropy. Have fun adjusting learning_rate (at the top
    of the file).
    For the different Tasks implement this file 4 times.
    1. like it is, just fix the issues stated in the comments.
    2. Change the optimizer to SGD + Nesterov momentum.
    3. Change 2. to also use L1 regression
    4. Change 3. to use L2 regression instead.
    For all those have fun playing with the parameters :D
"""
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

""" This operation is currently wrong, stop using global_step and do as in the code this was all based on.
"""
# Minimize using the loss calculated using squared error and l2_loss for the layers
train_op = optimizer.minimize(loss_op, global_step=global_step)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    loss_list = []
    y_list = []

    # Training cycle
    """ Implement the "mini-batches" aka batches like in the original file we based our code on.
        Best is probably to actually use what was done in the original.
        Make sure parameters are updated there ;)
    """
    for epoch in range(training_epochs):
        _, loss = sess.run([train_op, loss_op], feed_dict={X: mnist.train.images, Y: mnist.train.labels})
        if epoch % display_step == 0:
            loss_list.append(loss)
            y_list.append(epoch)
            print("Epoch:", '%04d' % (epoch + 1), "loss={:.9f}".format(loss))
    print("Optimization Finished!")

    # Test model
    # Apply softmax to logits. Determines the most likely label based on calculated confidence in the output layer
    pred = tf.nn.softmax(logits)
    # For all training samples test if the predicted value is equal to the actual label
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # For the current session or a default session, evaluate the network on given testdata and calculate the accuracy
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    plt.plot(loss_list)
    plt.show()
