
from __future__ import print_function
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# returns tuple (InterneuronWeights, weight)
# interneuronIndex is the layer we want to add an Interneuron to
# layers = an array of integers defining how many nodes per layer (including input and output layer)
def set_weights(layers, interneuronIndex):
    if(interneuronIndex == len(layers) - 1):
        #todo throw
        print("invalid interneuronIndex")
        return

    interneuronWeights = []
    weights = []
    for i in range(len(layers) - 1):
        tmp = tf.Variable(tf.random_normal([layers[i], layers[i+1]]))
        weights.append(tmp)

    for i in range(len(layers) - 2):
        tmp = tf.Variable(tf.random_normal([layers[i], layers[i+1]]))
        interneuronWeights.append(tmp)

    tmp = tf.Variable(tf.random_normal([layers[len(layers)-2], layers[interneuronIndex] * layers[interneuronIndex + 1]]))
    interneuronWeights.append(tmp)

    return (interneuronWeights, weights)

#like set_weights but with biases
def set_biases(layers, interneuronIndex):
    if(interneuronIndex == len(layers) - 1):
        #todo throw
        print("invalid interneuronIndex")
        return

    interneuronBiases = []
    biases = []
    for i in range(len(layers) - 1):
        tmp = tf.Variable(tf.random_normal([layers[i+1]]))
        biases.append(tmp)

    # len(layers) - 2 because i dont want to set the final layer
    for i in range(len(layers) - 2):
        tmp = tf.Variable(tf.random_normal([layers[i+1]]))
        interneuronBiases.append(tmp)

    tmp = tf.Variable(tf.random_normal([layers[interneuronIndex] * layers[interneuronIndex + 1]]))
    interneuronBiases.append(tmp)

    return (interneuronBiases, biases)

def create_model(x, weights, biases, interneuronWeights, interneuronBiases, interneuronIndex, layers):
    #TODO when do you do reduce sum?
    examples_size = tf.shape(x)[0]
    ident = tf.eye(examples_size)
    ident = tf.reshape(ident,[examples_size, examples_size, 1])


    #setup Interneurons
    previousLayer = tf.add(tf.matmul(x, interneuronWeights[0]), interneuronBiases[0])
    previousLayer = tf.nn.relu(previousLayer)
    for i in range(len(interneuronWeights) - 2): #-2 because first layer is already made and final layer is special
        nextLayer = tf.add(tf.matmul(previousLayer, interneuronWeights[i+1]), interneuronBiases[i+1]);
        nextLayer = tf.nn.relu(nextLayer)
        previousLayer = nextLayer

    interneuronOutputLayer = tf.add(tf.matmul(previousLayer, interneuronWeights[len(interneuronWeights)-1]), interneuronBiases[len(interneuronBiases)-1]);
    interneuronOutputLayer = tf.nn.relu(interneuronOutputLayer)
    interneuronOutputLayer = tf.reshape(interneuronOutputLayer, [examples_size, layers[interneuronIndex], layers[interneuronIndex+1]])

    #setup regular network
    previousLayer = tf.add(tf.matmul(x, weights[0]), biases[0])
    previousLayer = tf.nn.relu(previousLayer)
    for i in range(len(weights) - 2): #-2 because first layer is already made and final layer is special
        if (i + 1 == interneuronIndex):
            nextLayer = tf.matmul(tf.multiply(ident, previousLayer), tf.multiply(interneuronOutputLayer, weights[interneuronIndex]))
            nextLayer = tf.reduce_sum(nextLayer, axis=1)
            nextLayer = tf.add(nextLayer, biases[interneuronIndex])

            nextLayer = tf.nn.relu(nextLayer)
            previousLayer = nextLayer
        else:
            nextLayer = tf.add(tf.matmul(previousLayer, weights[i+1]), biases[i+1]);
            nextLayer = tf.nn.relu(nextLayer)
            previousLayer = nextLayer

    outputLayer = tf.matmul(previousLayer, weights[len(weights) - 1]) + biases[len(biases) - 1]
    return outputLayer


from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
with tf.device('/device:GPU:0'):  # Replace with device you are interested in
  bytes_in_use = BytesInUse()

#Set this to 1 in order to retrain the network
train_req = 0

#save
save_dest = os.getcwd() + "\Trained\model.ckpt"

# Parameters
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.001
decay_steps = 1.0
decay_rate = 0.5

training_epochs = 100
batch_size = 100
display_step = 1


node_structure = [784,256,256,10]

# tf Graph input
X = tf.placeholder("float", [None, node_structure[0]])
Y = tf.placeholder("float", [None, node_structure[-1]])

# Store layers weight & bias
interneuronWeights, weights = set_weights(node_structure, 1)

interneuronBiases, biases = set_biases(node_structure, 1)


# Create model
def Interneuron_MLP(x):
    return create_model(x, weights, biases, interneuronWeights, interneuronBiases, 1, node_structure)

# Construct model
logits = Interneuron_MLP(X)

saver = tf.train.Saver()

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    print("initial memory usage")
    print(sess.run(bytes_in_use))
    if train_req==1:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

        #Save the trained network
        save_path = saver.save(sess,save_dest)

    else:
        saver.restore(sess, save_dest)

    print("memory after loading model")
    print(sess.run(bytes_in_use))
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    testX = mnist.test.images[:1000,:]
    testY = mnist.test.labels[:1000]

    print("X len:")
    print(len(testX))

    print(sess.run(bytes_in_use))
    print("Accuracy1:", accuracy.eval({X: testX, Y: testY}))
    print(sess.run(bytes_in_use))
