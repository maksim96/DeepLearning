import numpy as np  
import tensorflow as tf
import matplotlib.pyplot as plt

# load data
data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

testY_one_hot = np.zeros((testY.size, 20))
testY_one_hot[np.arange(testY.size), testY] = 1

trainY_one_hot = np.zeros((trainY.size, 20))
trainY_one_hot[np.arange(trainY.size), trainY] = 1

print(trainX.shape)
print(trainY.shape)
# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input =  92*112
n_output = 20
 
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
 
# Store layers weight &amp; bias
weights = {
'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
'b1': tf.Variable(tf.random_normal([n_hidden_1])),
'b2': tf.Variable(tf.random_normal([n_hidden_2])),
'out': tf.Variable(tf.random_normal([n_output]))
}
 
# Construct model
pred = multilayer_perceptron(x, weights, biases)


# Define loss and optimizer
beta = 0.015
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y) + \
beta*(tf.nn.l2_loss(weights['h1']) +
    tf.nn.l2_loss(weights['h2']) +
    tf.nn.l2_loss(weights['out'])))
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                        100000, 0.96, staircase=True)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables

init = tf.global_variables_initializer()
loss_list = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for steps in range(1000):
        _, loss = sess.run([optimizer, cost], feed_dict={x: trainX, y: trainY_one_hot})
        if steps % 100 == 0:
            print(steps, loss)        

        loss_list.append(loss)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print("Test Accuracy:", accuracy.eval({x: testX, y: testY_one_hot}),  accuracy.eval({x: trainX, y: trainY_one_hot}))
    

plt.plot(loss_list)
plt.show()
