# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:52:12 2019

@author: Frank
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.interpolate import interp1d


# the convolutional neural network architecture
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolution 1: 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 6, 3)
        # Convolution 2: 6 input channel, 16 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(6, 16, 3)

        # linear connection, 20 different faces
        self.fc1 = nn.Linear(16 * 21 * 26, 20)

        ###################### Task 2.2 Dropout
        # for dropout usage, with p steer probabilty for :
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Max pooling 1 over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #  Max pooling 2 over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(1, 16 * 21 * 26)
        # linear layer
        x = self.fc1(x)
        return x


def map2image(cvweights):
    image = Image.new("L", (3, 3), "white")

    inter = interp1d([-1, 1], [0, 255])

    pixlist = []

    for i in range(3):
        for j in range(3):
            pixlist.append(int(float(inter(cvweights[i][j]))))
            #print(inter(cvweights[i][j]))

    image.putdata(pixlist)

    return image


# load data
data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']

print("The shape of input data: ", trainX.shape)
# example visualization, choose number betwenn 0 and 240 for image selection
im = np.reshape(trainX[0], (112, 92))
plt.imshow(im, cmap='gray')
plt.show()

# generate a network
net = Net()

# define the criterion or  error loss function
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
# create the optimizer as stochastic gradient descent  and set the learning rate
optimizer = optim.SGD(net.parameters(), lr=0.01)

# generate from the labels a target vector as a comparison for the output of the network
targetY = torch.zeros(trainY.shape[0], 20)
for i in range(0, trainY.shape[0]):
    # vector is 1, where the correct label is set, otherwise 0
    targetY[i][trainY[i]] = 1

# convert input values to numpy tensor
trainX_ten = torch.from_numpy(trainX)
# convert the shape of the input images, 1x1 for input to the network
# trainX_ten=trainX_ten.view(trainX.shape[0],112,92)
trainX_ten = trainX_ten.view(trainX.shape[0], 1, 1, 112, 92)
trainX_ten = trainX_ten.float()
# normalise inputs to range from 0 to 1
trainX_ten = trainX_ten / 255

# for testing
# reshape the test input for the network
testX_ten = torch.from_numpy(testX)
testX_ten = testX_ten.view(testX.shape[0], 1, 1, 112, 92)
testX_ten = testX_ten.float()
# normalise inputs to range from 0 to 1
testX_ten = testX_ten / 255

# for randomly selecting inputs
index = np.arange(240)
# saving the loss values for plotting them later
losses_list = []
clerror_list = []

# number of epochs
for ep in range(0, 30):
    # shuffle input indices
    np.random.shuffle(index)
    running_loss = 0

    for i in range(0, trainX.shape[0]):
        inp = trainX_ten[index[i]]
        # print(inp)
        # zero the gradient buffers
        optimizer.zero_grad()

        output = net(inp)
        # print(output)
        loss = criterion(output, targetY[index[i]])
        loss.backward()
        optimizer.step()  # Does the update

        # sum up losses to get the average running loss
        running_loss += loss.item()

    print("Epoch ", ep)
    # divide by number of training samples in this epoch
    running_loss = running_loss / trainX.shape[0]
    print("Loss : ", running_loss)
    losses_list.append(running_loss)

    # for classification error computation
    cor_pred = 0
    for i in range(0, testX.shape[0]):
        # compute the network output
        output = net(testX_ten[i])
        # print(output)
        # get the predicted class label as argmax
        cla = torch.argmax(output)
        # print(cla)
        if cla == testY[i]:  # correct prediction
            cor_pred += 1
    cor_pred = cor_pred / testX.shape[0]
    # as a percentage
    cor_pred *= 100
    print("Classification error: ", (100 - cor_pred), "%")
    clerror_list.append(100 - cor_pred)

# for visualization
losses = np.zeros((2, len(losses_list)))
for i in range(0, len(losses_list)):
    losses[0][i] = i
    losses[1][i] = losses_list[i]

clerror = np.zeros((2, len(clerror_list)))
for i in range(0, len(clerror_list)):
    clerror[0][i] = i
    clerror[1][i] = clerror_list[i]
# plot the loss function and classification error function
plt.plot(losses[0], losses[1])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.plot(clerror[0], clerror[1])
plt.xlabel("epoch")
plt.ylabel("classification error")
plt.show()

# final testing
cor_pred = 0
for i in range(0, testX.shape[0]):
    # compute the network output
    output = net(testX_ten[i])
    # print(output)
    # get the predicted class label as argmax
    cla = torch.argmax(output)
    # print(cla)
    if cla == testY[i]:  # correct prediction
        cor_pred += 1
cor_pred = cor_pred / testX.shape[0]
# as a percentage
cor_pred *= 100
print("Test accuracy: ", cor_pred, "%")

# extract the weights from convolution layer 1
c1weights = net.conv1.weight

# use PIL to create images from the weights
im1 = map2image(c1weights[0][0].detach().numpy())
im2 = map2image(c1weights[1][0].detach().numpy())
im3 = map2image(c1weights[2][0].detach().numpy())
im4 = map2image(c1weights[3][0].detach().numpy())
im5 = map2image(c1weights[4][0].detach().numpy())
im6 = map2image(c1weights[5][0].detach().numpy())

im1.resize((120, 120)).show()
im2.resize((120, 120)).show()
im3.resize((120, 120)).show()
im4.resize((120, 120)).show()
im5.resize((120, 120)).show()
im6.resize((120, 120)).show()
