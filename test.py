import os
import tensorflow as tf
import numpy as np


nodeStructure = [1,1,1,1]
iIndex = 2
biases = []
weights = []
ibiases = []
iweights = []

numINeurons = nodeStructure[iIndex] * nodeStructure[iIndex + 1]

#biases
for i in range(len(nodeStructure)):
    tmp = tf.Variable(tf.random_normal([nodeStructure[i+1]]))
    biases.append(tmp)

ibiases.append(tf.Variable(tf.random_normal( [numINeurons])))

#weights
for i in range(len(nodeStructure) - 1):
    tmp = tf.Variable(tf.random_normal([nodeStructure[i], nodeStructure[i+1]]))
    weights.append(tmp)

tmp = tf.Variable(tf.random_normal([self._nodeStructure[i], self._nodeStructure[i+1]]))

iweights.append( tf.Variable (tf.random_normal( [np.sum( nodeStructure), numINeurons]))

x = get_data()

#create model

examples_size = tf.shape(x)[0]
ident = tf.eye(examples_size)
ident = tf.reshape(ident,[examples_size, examples_size, 1])


#create interneurons
iinput = ?
interneuronOutputLayer = tf.add(tf.matmul(iinput, iweights[0]), ibiases[0])
interneuronOutputLayer = tf.nn.relu(interneuronOutputLayer)
interneuronOutputLayer = tf.reshape(interneuronOutputLayer, [examples_size, nodeStructure[iIndex], nodeStructure[iIndex+1]])
