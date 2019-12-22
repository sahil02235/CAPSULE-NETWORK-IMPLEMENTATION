# Capsle Network Implementation 

# Importing the libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)

# Import MNIST dataset
mnist = input_data.read_data_sets("data/mnist",one_hot=True)

# Batch size
batch_size = 50
epsilon = 1e-9

# Squash Function
def squash(sj):
    sj_norm = tf.reduce_sum(tf.square(sj), -2, keep_dims=True)
    scalar_factor = sj_norm / (1 + sj_norm) / tf.sqrt(sj_norm + epsilon)

    vj = scalar_factor * sj  

    return vj

# Dynamic Routing algorithm as stated in paper
def dynamic_routing(ui, bij, num_routing=10):
    
    #initialize weights wij by drawing from a random normal distribution
    wij = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32, 
                                  initializer=tf.random_normal_initializer(0.01))

    #initialize biases with a constant value
    biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))
    
    #define the primary capsules: (tf.tile replicates the tensor n times)
    ui = tf.tile(ui, [1, 1, 160, 1, 1])

    #compute the prediction vector
    u_hat = tf.reduce_sum(wij * ui, axis=3, keep_dims=True)
    
    #reshape the prediction vector
    u_hat = tf.reshape(u_hat, shape=[-1, 1152, 10, 16, 1])

    #stop gradient computation in the prediction vector
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    #perform dynamic routing for number of routing iterations
    for r in range(num_routing):
    
        #refer dynamic routing algorithm in the book for the detailed explanation on the following steps
        with tf.variable_scope('iter_' + str(r)):
               
            #step 1
            cij = tf.nn.softmax(bij, dim=2)
            
            #step 2
            if r == num_routing - 1:

                sj = tf.multiply(cij, u_hat)

                sj = tf.reduce_sum(sj, axis=1, keep_dims=True) + biases

                vj = squash(sj)

            elif r < num_routing - 1: 

                sj = tf.multiply(cij, u_hat_stopped)

                sj = tf.reduce_sum(sj, axis=1, keep_dims=True) + biases

                vj = squash(sj)

                vj_tiled = tf.tile(vj, [1, 1152, 1, 1, 1])

                coupling_coeff = tf.reduce_sum(u_hat_stopped * vj_tiled, axis=3, keep_dims=True)

                #step 3
                bij += coupling_coeff

    return vj

# Computing the primary capsules and Digits capsules
graph = tf.Graph()

with graph.as_default() as g:
     
    #placeholders for the input and output
    x = tf.placeholder(tf.float32, [batch_size, 784])
    y = tf.placeholder(tf.float32, [batch_size,10])
    
    #reshape the input x
    x_image = tf.reshape(x, [-1,28,28,1])

    #perform the convolutional operation and get the convolutional input,
    with tf.name_scope('convolutional_input'):
        input_data = tf.contrib.layers.conv2d(inputs=x_image, num_outputs=256, 
                                              kernel_size=9, padding='valid')
        
    
    #compute the primary capsules which extract the basic features such as edges.    
    #first, compute the capsules using convolution operation:
    capsules = []

    for i in range(8):

        with tf.name_scope('capsules_' + str(i)):
           
            #convolution operation
            output = tf.contrib.layers.conv2d(inputs=input_data, num_outputs=32,kernel_size=9,
                                              stride=2, padding='valid')
            
            #reshape the output
            output = tf.reshape(output, [batch_size, -1, 1, 1])
            
            #store the output which is capsule in the capsules list
            capsules.append(output)
    
    #concatenate all the capsules and form the primary capsule    
    primary_capsule = tf.concat(capsules, axis=2)
    
    #squash the primary capsule and get the probability i.e apply squash function and get the probability
    primary_capsule = squash(primary_capsule)
    

    #compute digit capsules using dynamic routing
    with tf.name_scope('dynamic_routing'):
        
        #reshape the primary capsule
        outputs = tf.reshape(primary_capsule, shape=(batch_size, -1, 1, primary_capsule.shape[-2].value, 1))
        
        #initialize bij with 0s
        bij = tf.constant(np.zeros([1, primary_capsule.shape[1].value, 10, 1, 1], dtype=np.float32))
        
        #compute the digit capsules using dynamic routing algorithm which takes 
        #the reshaped primary capsules and bij as inputs and returns the activity vector 
        digit_capsules = dynamic_routing(outputs, bij)
   
    digit_capsules = tf.squeeze(digit_capsules, axis=1)

# Masking the digit capsule
with graph.as_default() as g:
    with tf.variable_scope('Masking'):
        _
        # select the activity vector of given input image using the actual label y and mask out others
        masked_v = tf.multiply(tf.squeeze(digit_capsules), tf.reshape(y, (-1, 10, 1)))

# Defining the decoder
with graph.as_default() as g:
    
    with tf.name_scope('Decoder'):
        
        #masked digit capsule
        v_j = tf.reshape(masked_v, shape=(batch_size, -1))
        
        #first fully connected layer 
        fc1 = tf.contrib.layers.fully_connected(v_j, num_outputs=512)
           
        #second fully connected layer
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)

        #reconstructed image
        reconstructed_image = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

# Compute Accuracy
with graph.as_default() as g:
    with tf.variable_scope('accuracy'):
        
        #compute the length of each activity vector in the digit capsule 
        v_length = tf.sqrt(tf.reduce_sum(tf.square(digit_capsules), axis=2, keep_dims=True) + epsilon)
       
        #apply softmax to the length and get the probabilities
        softmax_v = tf.nn.softmax(v_length, dim=1)
       
        #select the index which got the highest probability this will give us the predicted digit 
        argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))    
        predicted_digit = tf.reshape(argmax_idx, shape=(batch_size, ))
        
        #compute the accuracy
        actual_digit = tf.to_int32(tf.argmax(y, axis=1))
        
        correct_pred = tf.equal(predicted_digit,actual_digit)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Loss
        
lambda_ = 0.5
alpha = 0.0005

with graph.as_default() as g:

    #margin loss
    max_left = tf.square(tf.maximum(0.,0.9 - v_length))
    max_right = tf.square(tf.maximum(0., v_length - 0.1))

    T_k = y
    
    #compute margin loss L_k for class k as given in (2)
    L_k = T_k * max_left + lambda_ * (1 - T_k) * max_right
    
    #compute total margin as given in refer equation (3)
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_k, axis=1))
    
    #reshape and get the original image
    original_image = tf.reshape(x, shape=(batch_size, -1))
    
    #compute reconstruction loss as shown in (4)
    squared = tf.square(reconstructed_image - original_image)
    reconstruction_loss = tf.reduce_mean(squared)

    #compute total loss which is the weighted sum of margin and reconstructed loss as shown in (5)
    total_loss = margin_loss + alpha * reconstruction_loss

with graph.as_default() as g:
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(total_loss)
    
num_epochs = 100
num_steps = int(len(mnist.train.images)/batch_size)

with tf.Session(graph=graph) as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)


    for epoch in range(num_epochs):
        for iteration in range(num_steps):
            batch_data, batch_labels = mnist.train.next_batch(batch_size)
            feed_dict = {x : batch_data, y : batch_labels}
     
            _, loss, acc = sess.run([optimizer, total_loss, accuracy], feed_dict=feed_dict)

            if iteration%10 == 0:
                print('Epoch: {}, iteration:{}, Loss:{} Accuracy: {}'.format(epoch,iteration,loss,acc))