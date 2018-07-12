import numpy as np
import math


'''This is a python module for a simple 1,0 neural network. A neural network closely represents the
graph data structure. A structure with nodes and edges between them. The neural network trains itself by
minimizing its loss through Forward Propagation and Back Propagation. With help from https://thecodacus.com/'''

#An edge between Nodes
class Edge:
    def __init__(self, connectedNode):
        self.connectedNode = connectedNode
        self.weight = np.random.normal()
        self.deltaWeight = 0.0

#A neuron or node
class Neuron:
    # Eta-Momentum, Alpha-Learning Rate
    # The learning rate is the size of the step that is taken in reducing the loss.
    # Momentum previous changes in the weights should influence the current direction of movement in weight space
    eta = 0.09
    alpha = 0.015

    def __init__(self,layer):
        self.dendrons =[]
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                edge = Edge(neuron)
                self.dendrons.append(edge)

    def addError(self,error):
        self.error = self.error + error

    #The sigmoid function
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x * 1.0))

    #The derivative of the sigmoid function
    def sigmoid_derivative(self,x):
        return x*(1.0-x)

    #Get and set methods
    def setError(self,error):
        self.error = error

    def setOutput(self,output):
        self.output = output

    def getOutput(self):
        return self.output

    #FORWARD PROPAGATION
    # check if there are any previously connected dendrons, if not then it’s an input or bias neuron.
    # Otherwise perform forward propagation by getting output of each connected neuron and multiplying it with
    #the connection weight and finally summing it up and passing it through an activation function.
    #summing them up and passing it through the activation function sigmoid and that the output.

    def forwardPropagation(self):
        output = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            output += dendron.connectedNode.getOutput()*dendron.weight
        self.output = self.sigmoid(output)

    def backPropagate(self):
        self.gradient = self.error*self.sigmoid_derivative(self.output)
        for dendron in self.dendrons:
            dendron.deltaWeight = Neuron.eta * (dendron.connectedNode.output * self.gradient) + self.alpha * dendron.deltaWeight
            dendron.weight = dendron.weight + dendron.deltaWeight
            dendron.connectedNode.addError(dendron.weight * self.gradient)
        self.error = 0;
#The Network class is a collection of Neurons and is the representation
# of the Neural Network

class Network:
    #build the skeleton of the network
    def __init__(self,structure):
        self.layers = []
        for numNeuron in structure:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers)==0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            #add a bias neuron
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def forwardPropagation(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.forwardPropagation()

    def backPropagation(self,target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    def train(self,inputs,outputs,eta,alpha):
        Neuron.eta = eta
        Neuron.alpha = alpha
        while True:
            err = 0
            for i in range(len(inputs)):
                self.setInput(inputs[i])
                self.forwardPropagation()
                self.backPropagation(outputs[i])
                err = err + net.getError(outputs[i])
            print("error: ", err)
            if err < 0.01:
                break

    #inputs are two dimensional
    def predict(self,inputs):
        self.setInput(inputs)
        self.forwardPropagation()
        return self.getResults()

topology = [2,3,2]
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
net = Network(topology)
net.train(inputs,outputs)