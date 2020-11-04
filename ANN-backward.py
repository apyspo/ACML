import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set(style='darkgrid')

# define the sigmoid function and derivatives sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))
def dersigmoid(a):
    return a*(1-a) 
# The training data and target
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
# Number of nodes in each layer 
input_dim = 8
hidden_dim = 3
output_dim = 8

# Initialize weight decay parameter lambda
weightDecay = 0  #0.0001, 0.0001, 0.001
learningCurve =[]
m = len(X)

start = time.time()

# Start training 
for learningRate in [0.1, 0.5, 1]:
    # Initialize random weights 
    W1 = np.random.randn(hidden_dim, input_dim)
    W2 = np.random.randn(output_dim, hidden_dim)
    # Initialize random bias term
    b1 = np.zeros((hidden_dim, 1))
    b2 = np.zeros((output_dim, 1))
    
    for i in range(1,500000):
        # forward
        l0 = X
        l1 = sigmoid(np.dot(W1, l0) +b1)
        l2 = sigmoid(np.dot(W2, l1) +b2)
        # backward
        l2_delta = (l2 - y)*dersigmoid(l2)
        l1_delta = np.dot(W2.T,l2_delta)*dersigmoid(l1)
        # Updata weights and bias
        gradientW2 = np.dot(l2_delta, l1.T)
        gradientW1 = np.dot(l1_delta, l0.T)
        gradientb2 = np.sum(l2_delta,axis=1,keepdims=True)
        gradientb1 = np.sum(l1_delta,axis=1,keepdims=True)
        W2 -= learningRate*(1/m * gradientW2 + weightDecay * W2)
        W1 -= learningRate*(1/m * gradientW1 + weightDecay * W1)
        b1 -= learningRate* 1/m * gradientb1
        b2 -= learningRate* 1/m * gradientb2
        # store the error for learning curve
        if (i%10000==0):
            learningCurve.append([i, learningRate, np.sum(1/2 * np.dot((l2 - y).T, l2 - y))])
            
print("Execution duration: " + str((time.time() - start)))
print("Output: ")
print(l2)
print("y: ")
print(y)
print("Output of hidden layer: ")
print(l1)
learningCurveDf = pd.DataFrame(learningCurve, columns=('currentIter','learningRate', 'MSE'))
sns.lineplot(x='currentIter', y='MSE', hue= 'learningRate',legend='full',data=learningCurveDf)
plt.show()