import numpy as np


def sigmoid_activation(x, takeDerivate = False):
    
    if(takeDerivate == True):
        return sigmoid_activation(x) * (1 - sigmoid_activation(x))
    
    return 1 / (1 + np.exp(-x))



def print_params(iter, x_train, expected_labels, weights, prediction):
    print("-----------------------------------------------------------\n")
    print("iteration # ", iter)
    print("Input data is... \n", x_train)
    print("Expected labels are... \n", expected_labels)
    print("Current weights are... \n", weights)
    print("Predictions for train data at this iteration are... \n", prediction)
    print("-----------------------------------------------------------\n")

def trainPerceptron(inputs, t, weights, rho, iterNo):
    
    dw = 0
    
    for iter in range(iterNo):
        for i in range(inputs.shape[0]):
        #write train code in here
            x = inputs[i,:]
            sum = weights * x
            y = sigmoid_activation(sum,False)
        #end your code
        dw = rho * (t - y) * (sigmoid_activation(y) * (1 - sigmoid_activation(y)))
        dw += dw
        if(iter % 10 == 0):
            print_params(iter, inputs, t, weights, y)
            
    
    return weights

def testPerceptron(sample_test, weights):

    y = 0   
    #write prediction code in here
    for i in range(len(sample_test) - 1):
        y += weights[i + 1] * sample_test[i]
        
  
    return y
 
#######our main code
np.random.seed(1)

from keras.datasets import cifar10    
(x_data, y_train), (x_test, y_test) = cifar10.load_data()


x_data = x_data.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

idx1 = np.array(np.where(y_train==0)).T
idx2 = np.array(np.where(y_train==1)).T
n1 = idx1.shape[0]
n2 = idx2.shape[0]
n = n1+n2
t = np.zeros((n,1), np.int32)
t[0:5000] = y_train[idx1[:,0]]
t[5000:10000] = y_train[idx2[:,0]]

x_train = np.zeros((n,3072), np.uint8)
x_train[0:5000,:] = x_data[idx1[:,0],:]
x_train[5000:10000,:] = x_data[idx2[:,0],:]

bias_x = np.ones((10000,1), np.uint8)
x_train = np.append(x_train, bias_x, axis=1) #add bias


weights = 2*np.random.random((3072,)) - 1

bias = np.ones((1,), np.float32)
weights = np.append(weights, bias, axis=0) #add bias
weights = weights.reshape(3073,)


##convert all variables to float
weights = np.float32(weights)
x_train = np.float32(x_train)/255
t = np.float32(t)
x_test = np.float32(x_test)/255
y_test = np.float32(y_test)
y_train = np.float32(y_train)

from sklearn.utils import shuffle
x_train, t = shuffle(x_train, t, random_state=0)


iterNo=100
rho = 0.1
weights = trainPerceptron(x_train, t, weights,rho, iterNo)

sample_test = x_test[6000,:]
sample_test = np.append(sample_test, bias, axis=0) #add bias

expected = y_test[3]
predicted = testPerceptron(sample_test, weights)


