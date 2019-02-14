
from __future__ import print_function
import os
from InterneuronModelGenerator import InterneuronModelGenerator

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

#Set this to 1 in order to retrain the network
train_req = 1

#save
save_dest = os.getcwd() + "\Trained\model.ckpt"

# Parameters
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.001
decay_steps = 1.0
decay_rate = 0.5

training_epochs = 100
batch_size = 100
display_step = 10

node_structure = [784,500,500,300,10]

# tf Graph input
X = tf.placeholder("float", [None, node_structure[0]])
Y = tf.placeholder("float", [None, node_structure[-1]])

# Create model
for v in range(len(node_structure) - 2):
    modelBuilder = InterneuronModelGenerator()
    interneuronIndex = v + 1
    modelBuilder.add_layer(node_structure[0], None)
    for i in range(len(node_structure) - 2):
        if (i + 1 == interneuronIndex):
            modelBuilder.add_interneuron_layer(node_structure[i + 1], None)
        else:
            modelBuilder.add_layer(node_structure[i + 1], None)

    # Construct model
    logits = modelBuilder.add_output_layer(node_structure[-1], X)

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

        t = 0
        for i in range(100):
            testX = mnist.test.images[(i * 100):((i+1) * 100),:]
            testY = mnist.test.labels[(i * 100):((i+1) * 100)]
            a = accuracy.eval({X: testX, Y: testY})
            t = t + a
            print('Accuracy %d: %f.'%(i+1,a))
        
        print('InterneuronIndex: %f Final Accuracy: %f.'%(v+1, t/10.0))
