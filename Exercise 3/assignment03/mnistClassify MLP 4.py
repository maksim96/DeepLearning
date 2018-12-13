from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("model_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 100
display_step = 1

# Network Parameters
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
def multilayer_perceptron(x, dropout_input, dropout_hidden):
    # Input dropout layer 20%
    dropout_0 = tf.nn.dropout(x, dropout_input)
    # Hidden fully connected layer with 800 neurons
    layer_1 = tf.add(tf.matmul(dropout_0, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # First hidden dropout layer 50%
    dropout_1 = tf.nn.dropout(layer_1, dropout_hidden)
    # Hidden fully connected layer with 800 neurons
    layer_2 = tf.add(tf.matmul(dropout_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Second hidden dropout layer 50%
    dropout_2 = tf.nn.dropout(layer_2, dropout_hidden)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(dropout_2, weights['out']) + biases['out']
    """ maybe we should implement here already the softmax layer,
        currently only used when predicting at the bottom of file
        and when doing the loss operation, not sure if completely
        correct, but we are using softmax in learning and prediction,
        so probably alright.
    """
    return out_layer


# Construct model
# using dropout placeholders for input and hidden layers
d_i = tf.placeholder(tf.float32, name="dropout_probability_input")
d_h = tf.placeholder(tf.float32, name="dropout_probability_hidden")
logits = multilayer_perceptron(X, d_i, d_h)

""" For this file implement:
        4: Change 3. to use L2 regression instead.
    
    Have fun playing with the parameters :D
"""
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# Minimize using the loss calculated using squared error and l2_loss for the layers
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    loss_list = []
    y_list = []

    # Training cycle
    """ Change batch size at top of the file.
    """
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_loss += loss / total_batch
        if epoch % display_step == 0:
            loss_list.append(avg_loss)
            y_list.append(epoch)
            print("Epoch:", '%04d' % (epoch + 1), "loss={:.9f}".format(avg_loss))
    print("Optimization Finished!")

    # Test model
    # Apply softmax to logits. Determines the most likely label based on calculated confidence in the output layer
    pred = tf.nn.softmax(logits)
    # For all training samples test if the predicted value is equal to the actual label
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # For the current session or a default session, evaluate the network on given testdata and calculate the accuracy
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, d_i: 1., d_h: 1.}))

    plt.plot(loss_list)
    plt.show()
