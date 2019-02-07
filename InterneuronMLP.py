
from __future__ import print_function
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf


#Set this to 1 in order to retrain the network
train_req = 0

#save
save_dest = os.getcwd() + "/Trained/model.ckpt"

# Parameters
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.001
decay_steps = 1.0
decay_rate = 0.5

training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

n_I1hidden_1 = 256 # 1st layer number of Interneurons
n_I1hidden_2 = 256 # 2nd layer number of Interneurons

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),

    #'I1h1': tf.Variable(tf.random_normal([n_input, n_I1hidden_1])),
    #'I1h2': tf.Variable(tf.random_normal([n_I1hidden_1, n_I1hidden_2])),
    #'Iout1': tf.Variable(tf.random_normal([n_I1hidden_2, n_input*n_hidden_1])),

    'I2h1': tf.Variable(tf.random_normal([n_input, n_I1hidden_1])),
    'I2h2': tf.Variable(tf.random_normal([n_I1hidden_1, n_I1hidden_2])),
    'Iout2': tf.Variable(tf.random_normal([n_I1hidden_2, n_hidden_1*n_hidden_2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes])),

    #'I1b1': tf.Variable(tf.random_normal([n_I1hidden_1])),
    #'I1b2': tf.Variable(tf.random_normal([n_I1hidden_2])),
    #'Iout1': tf.Variable(tf.random_normal([n_input*n_hidden_1])),

    'I2b1': tf.Variable(tf.random_normal([n_I1hidden_1])),
    'I2b2': tf.Variable(tf.random_normal([n_I1hidden_2])),
    'Iout2': tf.Variable(tf.random_normal([n_hidden_1*n_hidden_2]))
}


# Create model
def Interneuron_MLP(x):

    examples_size= tf.shape(x)[0]
    ident= tf.eye(examples_size)
    ident= tf.reshape(ident,[examples_size, examples_size, 1])

    # Hidden fully connected iLayer with 256 neurons
    #i1Layer_1 = tf.add(tf.matmul(x, weights['I1h1']), biases['I1b1'])
    #i1Layer_1 = tf.nn.relu(i1Layer_1)
    # Hidden fully connected ilayer with 256 neurons
    #i1Layer_2 = tf.add(tf.matmul(i1Layer_1, weights['I1h2']), biases['I1b2'])
    #i1Layer_2 = tf.nn.relu(i1Layer_2)
    # Output fully connected ilayer with a neuron for each neuron gate
    #iOut1_layer = tf.add(tf.matmul(i1Layer_2, weights['Iout1']), biases['Iout1'])
    #iOut1_layer = tf.nn.relu(iOut1_layer)
    #iOut1_layer = tf.reshape(iOut1_layer, [examples_size, n_input, n_hidden_1])

    # Hidden fully connected iLayer with 256 neurons
    i2Layer_1 = tf.add(tf.matmul(x, weights['I2h1']), biases['I2b1'])
    i2Layer_1 = tf.nn.relu(i2Layer_1)
    # Hidden fully connected ilayer with 256 neurons
    i2Layer_2 = tf.add(tf.matmul(i2Layer_1, weights['I2h2']), biases['I2b2'])
    i2Layer_2 = tf.nn.relu(i2Layer_2)
    # Output fully connected ilayer with a neuron for each neuron gate
    iOut2_layer = tf.add(tf.matmul(i2Layer_2, weights['Iout2']), biases['Iout2'])
    iOut2_layer = tf.nn.relu(iOut2_layer)
    iOut2_layer = tf.reshape(iOut2_layer, [examples_size, n_hidden_1, n_hidden_2])

    #layer_1 = tf.matmul(tf.multiply(ident, x), tf.multiply(iOut1_layer,weights['h1']))
    #layer_1 = tf.reduce_sum(layer_1,axis=1)
    #layer_1 = tf.add(layer_1, biases['b1'])

    #This is the traditional MLP layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.matmul(tf.multiply(ident, layer_1), tf.multiply(iOut2_layer,weights['h2']))
    layer_2 = tf.reduce_sum(layer_2, axis=1)
    layer_2 = tf.add(layer_2, biases['b2'])

    #This is the traditional MLP layer
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

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


    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy1:", accuracy.eval({X: mnist.test.images[:5000,:], Y: mnist.test.labels[:5000]}))
    print("Accuracy2:", accuracy.eval({X: mnist.test.images[5001:,:], Y: mnist.test.labels[5001:]}))