import numpy as np

def sigmoid(z):
  return 1/(1+np.exp(-z))
def dersigmoid(a):
    return a*(1-a) 

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

W1 = np.random.normal(0,0.01,size=(8,2))
W2 = np.random.normal(0,0.01,size=(2,8))
b1 = np.random.normal(0,0.01, size=(8,2))
b2 = np.random.normal(0,0.01, size=(8,8))
for j in range(1,1000000):
    l0 = X
    l1 = sigmoid(np.dot(l0, W1)+b1)
    l2 = sigmoid(np.dot(l1, W2)+b2)
    l2_error = y - l2
    l2_delta = l2_error*dersigmoid(l2)
    l1_error = l2_delta.dot(W2.T)
    l1_delta = l1_error*dersigmoid(l1)  
    W2 += 0.01*l1.T.dot(l2_delta)
    W1 += 0.01*l0.T.dot(l1_delta)
    b1 += 0.01*l1_delta
    b2 += 0.01*l2_delta
print(l2)
print(y)