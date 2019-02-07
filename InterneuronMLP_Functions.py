# returns tuple (InterneuronWeights, weight)
# interneuronIndex is the layer we want to add an Interneuron to
# layers = an array of integers defining how many nodes per layer (including input and output layer)
def set_weights(layers, interneuronIndex):
    if(interneuronIndex == len(layers) - 1)
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
    if(interneuronIndex == len(layers) - 1)
        #todo throw
        print("invalid interneuronIndex")
        return

    interneuronBiases = []
    biases = []
    for i in range(len(layers) - 1):
        tmp = tf.Variable(tf.random_normal([layers[i+1]))
        biases.append(tmp)

    # len(layers) - 2 because i dont want to set the final layer
    for i in range(len(layers) - 2)
        tmp = tf.Variable(tf.random_normal(layers[i+1]]))
        interneuronBiases.append(tmp)

    tmp = tf.Variable(tf.random_normal(layers[interneuronIndex] * layers[interneuronIndex + 1]]))
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
        if (i + 1 == interneuronIndex)
            nextLayer = tf.matmul(tf.multiply(ident, previousLayer), tf.multiply(interneuronOutputLayer, weights[interneuronIndex]))
            nextLayer = tf.reduce_sum(nextLayer, axis=1)
            nextLayer = tf.add(nextLayer, biases[interneuronIndex])

            nextLayer = tf.nn.relu(nextLayer)
            previousLayer = nextLayer
        else
            nextLayer = tf.add(tf.matmul(previousLayer, weights[i+1]), biases[i+1]);
            nextLayer = tf.nn.relu(nextLayer)
            previousLayer = nextLayer

    outputLayer = tf.matmul(previousLayer, weights[len(weights) - 1]) + biases[len(biases) - 1]
    return outputLayer


























print("stop deleting my newlines")
