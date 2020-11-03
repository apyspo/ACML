import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# fSigmoid * (1 - fSigmoid)
def dersigmoid(a):
    return a * (1-a)

y = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 1]
        ])

# Original training set
X = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 0, 0, 1]
        ])

# Training subset with only 6 / 8
XS = np.array([
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

# 8 training sample, use multiple time same samples
#first layer out 3x6 = 3x8 x 8x6
#second layer out 8x1 = 8x4 (after adding bias) x 4x1
N_HIDDEN_NODES = 3
N_INPUT_FEATURES = 8 # Add 1 for BIAS X0
N_OUTPUT = 8
#W1=np.random.normal(0,0.01, size=(N_HIDDEN_NODES, N_INPUT_FEATURES))
#W2=np.random.normal(0,0.01, size=(N_OUTPUT, N_HIDDEN_NODES))
W1=np.random.normal(0,0.01, size=(3, 9))
W2=np.random.normal(0,0.01, size=(8, 4))

### Initialize training and test set arrays, Bias
B = np.array([
    [1]
])

B1A = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1]
])

B2A = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1]
])

weightDecay = 0.001
learningRate = 0.1

ITER=400000
for currentIter in range(1, ITER):
    i = currentIter % 9
    if i != 0:
        #print(i)
        #X1 = X[:, i-1:i]
        # adding bias
        #X1 = np.concatenate((B, X1), axis=0)
        X1 = np.concatenate((B1A, X), axis=0)
        #y1 = X[:, i-1:i]
        Z2 = np.dot(W1, X1)
        L1Out = sigmoid(Z2)
        #print("Out First Layer :" + str(L1Out))

        # For second layer
        L1Out = np.concatenate((B2A, L1Out), axis=0)
        Z3 = np.dot(W2, L1Out)
        L2Out = sigmoid(Z3)
        #print("Out Second Layer :" + str(L2Out))

        L2Error = L2Out - X #y1 #y1 - L2Out
        L2Delta = L2Error * dersigmoid(L2Out)
        #L2Delta = L2Delta[1:, :]
        L1Error =  np.dot(W2.T, L2Delta)
        L1Delta = L1Error * dersigmoid(L1Out)
        W2 -= learningRate*(np.dot(L2Delta, L1Out.T) + weightDecay * W2) 
        L1Delta = L1Delta[1:, :]
        W1 -= learningRate*(np.dot(L1Delta, X1.T) + weightDecay * W1)

print("After training prediction performance :")
#print(y1 - L2Out)
print(X)
print(L2Out)
print(sigmoid(np.dot(W1, X1)))


# Use trained weights for testing
#X1 = X[:, 1:2]
#X1 = np.concatenate((B, X1), axis=0)
#y1 = X[:, 1:2]
#Z2 = np.dot(W1, X1)
#L1Out = sigmoid(Z2)
#L1Out = np.concatenate((B, L1Out), axis=0)
#Z3 = np.dot(W2, L1Out)
#L2Out = sigmoid(Z3)

#print(y1)
#print(L2Out)