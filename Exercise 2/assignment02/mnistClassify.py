from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("model_data/", one_hot=True)

# Parameters
# learning_rate = 0.01
training_epochs = 300
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
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

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# Calculate the loss based on the squared error of the output layer and the l2 loss for all weights
beta = 0.5
loss_op = tf.reduce_mean(
    tf.losses.mean_squared_error(labels=Y, predictions=tf.nn.softmax(logits=logits), weights=5000.0) +
    beta * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out'])))

# Define learning with exponential decay for gradient descent trainer
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.015
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 10000, 0.80)

# Train using gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# Minimize using the loss calculated using squared error and l2_loss for the layers
train_op = optimizer.minimize(loss_op, global_step=global_step)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    loss_list = []
    y_list = []

    # Training cycle
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
