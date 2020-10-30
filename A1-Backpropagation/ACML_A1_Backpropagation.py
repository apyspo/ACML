import numpy as np
from numpy import random
import math

def fSigmoid(z):
    return 1 / (1 + np.exp(-z))

# fSigmoid * (1 - fSigmoid)
def derfSigmoid(z):
    return  (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))) )

# 8 training sample, use multiple time same samples
#first layer 3x1 = 3x9 x 9x1
#second layer 8x1 = 8x4 x 4x1
N_HIDDEN_NODES = 3
N_INPUT_FEATURES = 8 # Add 1 for BIAS X0
N_OUTPUT = 8
#W1=np.random.rand(N_HIDDEN_NODES, N_INPUT_FEATURES + 1)
#W2=random.rand(N_OUTPUT, N_HIDDEN_NODES + 1)
W1=np.random.rand(N_HIDDEN_NODES, N_INPUT_FEATURES)
W2=random.rand(N_OUTPUT, N_HIDDEN_NODES)


### Initialize training and test set arrays, Bias
B = np.array([
    [1]
])


Y = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0]
        ])

# Original training set
training_set = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0]
        ])

# Training subset with only 6 / 8
training_subset = np.array([
        [0, 0, 0, 0, 0, 0], 
        [1, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 0, 0]
        ])

### End of Array Initialization

#print(range(1, training_set1.shape[1]+1))
T2 = np.zeros((N_OUTPUT, N_HIDDEN_NODES + 1))
ITER=10000

for currentIter in range(1, ITER):
    i = currentIter % 7
    if i != 0:
        #print(i)
        X1 = training_set[:, i-1:i]
        #X1 = np.concatenate((B, X1), axis=0)
        Y1 = training_set[:, i-1:i]
        Z2 = np.dot(W1, X1)
        A2 = fSigmoid(Z2)
        #print("Out First Layer :")
        #print(A2)
        # Out First Layer

        # For second layer
        #A2 = np.concatenate((B, A2), axis=0)
        Z3 = np.dot(W2, A2)
        H3 = fSigmoid(Z3)
        #print("Out Second Layer :")
        #print(H3)
        #print("Error:")
        #print(Y1 - H3)

        Error = Y1 - H3
        Delta = Error * derfSigmoid(H3)

        Z2Error =  np.dot(W2.T, Delta)
        Z2Delta = Z2Error * derfSigmoid(A2)
        W1 = W1 + np.dot(Z2Delta, X1.T)
        W2 = W2 + np.dot(Delta, A2.T)

        # To check sum of errors for trend during training
        print(np.sum(Y1-H3))

print("After training prediction performance :")
print(Y1 - H3)
print(Y1)
print(H3)

# Use trained weights for testing
X1 = training_set[:, 0:1]
Y1 = training_set[:, 0:1]
Z2 = np.dot(W1, X1)
A2 = fSigmoid(Z2)
Z3 = np.dot(W2, A2)
H3 = fSigmoid(Z3)

print(H3)
print(Y1)