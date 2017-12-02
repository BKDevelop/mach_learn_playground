# 3-layer neural network

import numpy as np

# sigmoid function (s-shape function)
# sig(t) = 1/(1 + e^(-t) )
## this function is used to activate artifical neurons. Derivable functions are needed to perform machine-learning algorithms like back-propagation. Sigmoid function is chosen because it is very easy to deviate (sig'(x)=sig(x)(1-sig(x)))

def sig(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    
    return 1/(1+np.exp(-x))
###############################

# some input and output data
# input matrix
x = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

#output matrix
y = np.array([[0],
             [1],
             [1],
             [0]])

# fix seed for random numbers -> get the same set of random numbers every time -> easier testing
np.random.seed(1)

##############################

# Synapses
## connect each neuron of one to layer to all neurons of next layer
## two arrays, 3-by-4 and 4-by-1, analogous to input and output dataset
## assign random weights
## 2*np.random.random((x,y) - 1 -> x-by-y array with random numbers from [-1,1)
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1 

###############################

# train the network (60000 training iterations)
for i in range(60000):
    
    # first layer: input data
    layer0 = x
    
    # second + third: perform dot product matrix operation for previous layer and weights -> calculating forward through the network
    
    layer1 = sig(np.dot(layer0,syn0))
    #layer2 is predicted output
    layer2 = sig(np.dot(layer1,syn1))
    
    # compare to expected output
    layer2_error = y - layer2
    
    # print mean error rate every 10000 steps to see if it goes down
    if(i % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(layer2_error))))
    
    # use Sigmoid function to derive layer2 data, multiply with error rate          
    layer2_delta = layer2_error * sig(layer2, deriv=True)
    
    # Backpropagation: 
    # see how much layer1 contributed to the error --> multiply layer2_delta with transposed weights (syn1) matrix              
    layer1_error = layer2_delta.dot(syn1.T)
    
    # build layer1_delta analogous to layer2
    
    layer1_delta = layer1_error*sig(layer1,deriv=True)
    
    # gradient descent
    # use deltas to update the weights (-> reduce error) every iteration
    
    syn1 = layer1.T.dot(layer2_delta)
    syn0 = layer0.T.dot(layer1_delta)
              
##########################################

# Results!
            
print("Output after training")
print(layer2)