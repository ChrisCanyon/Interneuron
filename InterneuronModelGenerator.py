import os
import tensorflow as tf

class InterneuronModelGenerator:
    def __init__(self):
        self._interneuronIndex = None
        self._weights = []
        self._biases = []
        self._nodeStructure = []
        self._nodeActivations = [] #currently unused
        self._interneuronWeights = []
        self._interneuronBiases = []
        return

    def add_layer(self, nodes, activation):
        self._nodeStructure.append(nodes)
        self._nodeActivations.append(activation)
        return 

    #currently only supports one interneuron layer
    def add_interneuron_layer(self, nodes, activation):
        self._nodeStructure.append(nodes)
        self._nodeActivations.append(activation)
        self._interneuronIndex = len(self._nodeStructure) - 1
        return

    def add_output_layer(self, nodes, x):
        self._nodeStructure.append(nodes)
        self._set_weights()
        self._set_biases()        
        return self._create_model(x)

    def _set_weights(self):
        if(self._interneuronIndex == len(self._nodeStructure) - 1):
            #todo throw
            print("invalid self._interneuronIndex")
            return

        # Create main network weights
        for i in range(len(self._nodeStructure) - 1):
            tmp = tf.Variable(tf.random_normal([self._nodeStructure[i], self._nodeStructure[i+1]]))
            self._weights.append(tmp)

        # Create interneuron network weights
        for i in range(len(self._nodeStructure) - 2):
            tmp = tf.Variable(tf.random_normal([self._nodeStructure[i], self._nodeStructure[i+1]]))
            self._interneuronWeights.append(tmp)

        # create interneuron network output weights
        tmp = tf.Variable(tf.random_normal([self._nodeStructure[len(self._nodeStructure)-2], self._nodeStructure[self._interneuronIndex] * self._nodeStructure[self._interneuronIndex + 1]]))
        self._interneuronWeights.append(tmp)

        return (self._interneuronWeights, self._weights)

    #like set_weights but with biases
    def _set_biases(self):
        if(self._interneuronIndex == len(self._nodeStructure) - 1):
            #todo throw
            print("invalid self._interneuronIndex")
            return

        # create main network biases
        for i in range(len(self._nodeStructure) - 1):
            tmp = tf.Variable(tf.random_normal([self._nodeStructure[i+1]]))
            self._biases.append(tmp)

        # create interneuron network biases
        for i in range(len(self._nodeStructure) - 2):
            tmp = tf.Variable(tf.random_normal([self._nodeStructure[i+1]]))
            self._interneuronBiases.append(tmp)
        
        # create interneuron network output biases
        tmp = tf.Variable(tf.random_normal([self._nodeStructure[self._interneuronIndex] * self._nodeStructure[self._interneuronIndex + 1]]))
        self._interneuronBiases.append(tmp)

        return (self._interneuronBiases, self._biases)

    def _create_model(self, x):
        #TODO when do you do reduce sum?
        examples_size = tf.shape(x)[0]
        ident = tf.eye(examples_size)
        ident = tf.reshape(ident,[examples_size, examples_size, 1])


        #setup Interneurons
        previousLayer = tf.add(tf.matmul(x, self._interneuronWeights[0]), self._interneuronBiases[0])
        previousLayer = tf.nn.relu(previousLayer)
        for i in range(len(self._interneuronWeights) - 2): #-2 because first layer is already made and final layer is special
            nextLayer = tf.add(tf.matmul(previousLayer, self._interneuronWeights[i+1]), self._interneuronBiases[i+1])
            nextLayer = tf.nn.relu(nextLayer)
            previousLayer = nextLayer

        interneuronOutputLayer = tf.add(tf.matmul(previousLayer, self._interneuronWeights[len(self._interneuronWeights)-1]), self._interneuronBiases[len(self._interneuronBiases)-1])
        interneuronOutputLayer = tf.nn.relu(interneuronOutputLayer)
        interneuronOutputLayer = tf.reshape(interneuronOutputLayer, [examples_size, self._nodeStructure[self._interneuronIndex], self._nodeStructure[self._interneuronIndex+1]])

        #setup regular network
        previousLayer = tf.add(tf.matmul(x, self._weights[0]), self._biases[0])
        previousLayer = tf.nn.relu(previousLayer)
        for i in range(len(self._weights) - 2): #-2 because first layer is already made and final layer is special
            if (i + 1 == self._interneuronIndex):
                nextLayer = tf.matmul(tf.multiply(ident, previousLayer), tf.multiply(interneuronOutputLayer, self._weights[self._interneuronIndex]))
                nextLayer = tf.reduce_sum(nextLayer, axis=1)
                nextLayer = tf.add(nextLayer, self._biases[self._interneuronIndex])

                nextLayer = tf.nn.relu(nextLayer)
                previousLayer = nextLayer
            else:
                nextLayer = tf.add(tf.matmul(previousLayer, self._weights[i+1]), self._biases[i+1]);
                nextLayer = tf.nn.relu(nextLayer)
                previousLayer = nextLayer

        outputLayer = tf.matmul(previousLayer, self._weights[len(self._weights) - 1]) + self._biases[len(self._biases) - 1]
        return outputLayer

