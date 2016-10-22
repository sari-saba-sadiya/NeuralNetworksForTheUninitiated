
# coding: utf-8

######################################################
# Created by Sari Saba Sadiya Winter 2016
# License Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# Everyone is allowed to copy, remix, and redistribue,
# but commercial use is not allowed
#
# This is meant to be an example of a simple back 
# propagation based Neural Network. due to efficiency 
# issues I strongly discourage the use of this code
# for any non pedagogical goals.
#######################################################

import math
import random
import numpy
import scipy.io
from array import array

########################################################
# Define the functions that can be used for activation
# and auxilary functions
########################################################
def linear(x):
    return x

def linearGrad(x):
    return 1

def lossFunc(y1,y2):
    if len(y1) != len(y2) :
        print("ERROR: size mismatch between y1 and y2")
        raise NameError('losFuncError')
    return sum(math.pow(y1[i]-y2[i],2) for i in range(0,len(y1)))

def toyLossFunc(y1,y2):
    if len(y1) != len(y2) :
        print("ERROR: size mismatch between y1 and y2")
        raise NameError('losFuncError')
    return (sum(math.pow(y1[i]-y2[i],2) for i in range(0,len(y1))))/2

def sigmoidFunction(x):
    return 1/(1+math.exp(-x))

def sigmoidFunctionGrad(x):
    return sigmoidFunction(x)*(1-sigmoidFunction(x))

def tahnFunction(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def tahnFunctionGrad(x):
    return (1-math.pow(tahnFunction(x),2))

########################################################
# Neural Netwok Implementation
########################################################
### Neuron
class Neuron:
    def __init__(self,weights,bias,activation_func):
        #initialize weights from i to our neuron (j)
        self.activation_func = activation_func
        self.weights = [weights[i] for i in range(0,len(weights))]
        self.bias = bias
        
    def net(self,input_array):
        #calculate the pre-activation of the neuron
        #input array contains the activation of the previous function
        net = [input_array[i]*self.weights[i] for i in range(0,len(input_array))]
        net.append(self.bias)
        return sum(net)

    def out(self,input_array):
        #calculate the activation of the neuron
        #input array contains the activation of the previous function
        net = self.net(input_array)
        return self.activation_func(net),net

### NeuronLayer
class NeuronLayer:
    def __init__(self,weights,biases,activation_func):
        #initialize a hidden layer initialize neuron j
        self.neuron_array = [Neuron(weights[j],biases[j],activation_func) for j in range(0,len(weights))]
    
    def out(self,input_array):
        #input array contains the activation of the previous function
        out_array = []
        net_array = []
        for neuron in self.neuron_array:
            out,net = neuron.out(input_array)
            out_array.append(out)
            net_array.append(net)
        return out_array,net_array
    
    def weight(self,i,j):
        return self.neuron_array[j].weights[i]


### NeuralNetwork
class NeuralNetwork:
    def __init__(self,n_hidden_layers,weights,biases,activation_func,out_func):
        if len(weights) !=  n_hidden_layers+1:
            print("ERROR: size mismatch, Weights Array length should be number of hidden layers + 1")
            raise NameError('NeuralNetworkError')        
        #weights are an array W[k][i][j]  of the weights from neuron i to j at layer k
        self.n_hidden = n_hidden_layers
        self.layers = []
        #initialize hidden layers
        for k in range(0,self.n_hidden):
            h_layer_k = NeuronLayer(weights[k][0].T,biases[k][0],activation_func)
            self.layers.append(h_layer_k)
        #initialize output layer
        out_layer = NeuronLayer(weights[-1][0].T,biases[-1][0],out_func)
        self.layers.append(out_layer)
        
    def forwordPass(self,input_array):
        activations = [input_array]
        nets = [0]
        for layer in self.layers:
            #Last Layer's output are this ones input
            output,net = layer.out(activations[-1])
            activations.append(output)
            nets.append(net)
        return activations,nets
    
    def backwordPass(self,target,activations,nets,activation_func_grad,out_func_grad):
            Wgrads = [] # the gradients by W for each layer
            Ograds = [] # the gradients by the "out" of the previous layer
            Bgrads = [] # the gradients by b for each layer
            for k in range(len(self.layers)-1,-1,-1):
                k1 = k+1 #becuase the activation vector has an extra
                if k == len(self.layers)-1:
                    #calculate the gradient for w_i_j
                    dW = []
                    #calculate the gradient E^k_j / out^(k-1)_i
                    dO = []
                    #calculate the gradient for b
                    dB = []
                    for i in range(0,len(self.layers[k-1].neuron_array)):
                        dWi = []
                        dOi = []
                        for j in range(0,len(self.layers[k].neuron_array)):
                            # Becuase the loss function is (||target-y||_2)^2 we get the derivative
                            # -2*(target[j]-activations[k][j])
                            dWij = -2*(target[j]-activations[k1][j])*out_func_grad(nets[k1][j])*activations[k1-1][i]
                            dOij = -2*(target[j]-activations[k1][j])*out_func_grad(nets[k1][j])*self.layers[k].weight(i,j)
                            dWi.append(dWij)
                            dOi.append(dOij)
                            if i==0:
                                dBj = -2*(target[j]-activations[k1][j])*out_func_grad(nets[k1][j])
                                dB.append(dBj)
                        dW.append(dWi)
                        dO.append(sum(dOi))
                    Wgrads.append(dW)
                    Ograds.append(dO)
                    Bgrads.append(dB)
                else: #calculation for Hidden layers
                    #calculate the gradient for w_i_j
                    dW = []
                    #calculate the gradient E^k_j / out^(k-1)_i
                    dO = []
                    #calculate the gradient for b
                    dB = []
                    for i in range(0,len(self.layers[k-1].neuron_array)):
                        dWi = []
                        dOi = []
                        for j in range(0,len(self.layers[k].neuron_array)):
                            dWij = Ograds[-1][j]*activation_func_grad(nets[k1][j])*activations[k1-1][i]
                            dOij = Ograds[-1][j]*activation_func_grad(nets[k1][j])*self.layers[k].weight(i,j)
                            dWi.append(dWij)
                            dOi.append(dOij)
                            if i==0:
                                dBj = Ograds[-1][j]*activation_func_grad(nets[k1][j])
                                dB.append(dBj)
                        dW.append(dWi)
                        dO.append(sum(dOi))
                    Wgrads.append(dW)
                    Ograds.append(dO)
                    Bgrads.append(dB)
            return Wgrads[::-1],Bgrads[::-1]
    
    def printWeights(self):
        for k in range(len(self.layers)):
            for j in range(len(self.layers[k].neuron_array)):
                for i in range(len(self.layers[k].neuron_array)):
                    print(k,i,j," w ",self.layers[k].neuron_array[i].weights[j])

########################################################
# Check results via numerical calculations
########################################################

def gradientCheck(n_hidden_layers,x,y,w,b,activation_func,activation_func_grad,thershold=1e-5,h=1e-6):
    nn = NeuralNetwork(9,w,b,activation_func,linear)
    activations,nets = nn.forwordPass(x)
    loss = lossFunc(y,activations[-1])
    Wgrads,Bgrads = nn.backwordPass(y,activations,nets,activation_func_grad,linearGrad)
    for k in range(len(w)-1,-1,-1):
        for i in range(0,len(w[k][0])):
            for j in range(0,len(w[k][0][i])):
                w[k][0][i][j] = w[k][0][i][j] + h
                nnh = NeuralNetwork(9,w,b,activation_func,linear)
                activationsh,netsh = nnh.forwordPass(x)
                lossh = lossFunc(y,activationsh[-1])
                numGrad = (lossh-loss)/h
                diff = abs(numGrad-Wgrads[k][i][j])
                #print(diff) Biggest diss is ~7.5e-06
                if(diff>thershold):
                    print("check fail, diff is: ",diff,"at ",k,i,j)
                    print("analytical grad is: ",Wgrads[k][i][j])
                    print("numerical grad is:",numGrad)
                    print("activation and output (and prev activation):",activations[k+1][j],y[j],activations[k][i])
                    print("net in and weight:",nets[k+1][j],nnh.layers[k].weight(i,j))
                    raise NameError("gradientCheckFail")                   
                w[k][0][i][j] = w[k][0][i][j] - h
                if (i==0):
                    b[k][0][j] = b[k][0][j] + h
                    nnh = NeuralNetwork(9,w,b,activation_func,linear)
                    activationsh,netsh = nnh.forwordPass(x)
                    lossh = lossFunc(y,activationsh[-1])
                    numGrad = (lossh-loss)/h
                    diff = abs(numGrad-Bgrads[k][j])
                    if(diff>thershold):
                        print("bias check fail, diff is: ",diff,"at ",k,i,j)
                        print("analytical bias grad is: ",Wgrads[k][i][j])
                        print("numerical bias grad is:",numGrad)
                        print("activation and output (and prev activation):",activations[k+1][j],y[j],activations[k][i])
                        print("net in and weight:",nets[k+1][j],nnh.layers[k].weight(i,j))
                        raise NameError("gradientCheckFail")    
                    b[k][0][j] = b[k][0][j] - h

########################################################
# HOW TO RUN
########################################################

mat = scipy.io.loadmat('input.mat')
w = mat['W']
x = mat['x']
b = mat['b']
y = mat['y']

#Initialize the network
nn = NeuralNetwork(9,w,b,sigmoidFunction,linear)

#Run forward algorithm with the input and output desired
activations,nets = nn.forwordPass(x[0])

#Run backward algorithm
Wgrads,Bgrads = nn.backwordPass(y[0],activations,nets,sigmoidFunctionGrad,linearGrad)


#Run gradient check, calculate gradients numerically and compare 
gradientCheck(9,x[0],y[0],w,b,sigmoidFunction,sigmoidFunctionGrad)


#calculate the gradients for each x and y
allWgrad = []
allBgrad = []
for i in range(0,len(x1)):
    activations,nets = nn.forwordPass(x[i])
    Wgrads,Bgrads = nn.backwordPass(y[i],activations,nets,sigmoidFunctionGrad,linearGrad)
    allWgrad.append(Wgrads)
    allBgrad.append(Bgrads)

########################################################
# Save gradients into files
########################################################

# Save bias Gradient into b_gredients.mat
b_gradients = numpy.empty((len(allBgrad),), dtype=numpy.object)
for i in range(0,len(allBgrad)):
    b_gradients[i] = allBgrad[i]

scipy.io.savemat("b_gredients.mat", {"b_gradients":b_gradients})
# Save weights Gradient into b_gredients.mat
w_gradients = numpy.empty((len(allWgrad),), dtype=numpy.object)
for t in range(0,len(allWgrad)):
    w_gradients_t = numpy.empty((len(allWgrad[t]),), dtype=numpy.object)
    for k in range(0,len(allWgrad[t])):
        w_gradients_t_k = numpy.empty((len(allWgrad[t][k]),), dtype=numpy.object)
        for i in range(0,len(allWgrad[t][k])):
            w_gradients_t_k[i] = allWgrad[t][k][i]
        w_gradients_t[k]=w_gradients_t_k
    w_gradients[t] = w_gradients_t
scipy.io.savemat("w_gredients.mat", {"w_gradients":w_gradients})




