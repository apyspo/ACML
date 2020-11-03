import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
weightDecay = 0.001
learningCurve =[]

# Start training
for learningRate in [0.01, 0.05, 0.1]:
    
    # Initialize random weights 
    W1 = np.random.normal(0,0.01,size=(input_dim, hidden_dim))
    W2 = np.random.normal(0,0.01,size=(hidden_dim, output_dim))
    # Initialize random bias term
    b1 = np.random.normal(0,0.01, size=(input_dim, hidden_dim))
    b2 = np.random.normal(0,0.01, size=(input_dim, output_dim))
    
    for i in range(1,50000):
        # forward
        l0 = X
        l1 = sigmoid(np.dot(l0, W1)+b1)
        l2 = sigmoid(np.dot(l1, W2)+b2)
        # backward
        l2_error = l2 - y
        l2_delta = l2_error*dersigmoid(l2)
        l1_error = l2_delta.dot(W2.T)
        l1_delta = l1_error*dersigmoid(l1)
        # Updata weights and bias
        W2 -= learningRate*(l1.T.dot(l2_delta) + weightDecay * W2)
        W1 -= learningRate*(l0.T.dot(l1_delta) + weightDecay * W1)
        b1 -= learningRate*l1_delta
        b2 -= learningRate*l2_delta
        # store the error for learning curve
        if (i%1000==0):
            learningCurve.append([i, learningRate, np.sum(1/2 * np.dot(l2_error.T, l2_error))])
            

print(l2)
print(y)
learningCurveDf = pd.DataFrame(learningCurve, columns=('currentIter','learningRate', 'MSE'))
sns.lineplot(x='currentIter', y='MSE', hue= 'learningRate',legend='full',data=learningCurveDf)
plt.show()
