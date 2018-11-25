import numpy
import scipy.special
import matplotlib.pyplot

class artificialNeuralNetwork :
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
        self.inputNodes =inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate   
        self.activationFunction = lambda x : scipy.special.expit(x)     
        #● A matrix for the weights for links between the input and hidden layers,
        #  W​ input_hidden​ , of size (​ hidden_nodes​ by input_nodes)​. 
        #● And another matrix for the links between the hidden and output layers,
        #  Whidden_output​ , of size (​ output_nodes​ by hidden_nodes)​.
        self.weightsInputHidden = numpy.random.normal(0,self.hiddenNodes**-0.5,(self.hiddenNodes,self.inputNodes))
        self.weightsHiddenOutput = numpy.random.normal(0,self.outputNodes**-0.5,(self.outputNodes,self.hiddenNodes))

    def setLearningRate(self,newLearningRate):
        self.learningRate = newLearningRate

    def train(self,inputs_list,targets_list):
        # convert inputs list to 2d array 
        inputs = numpy.array(inputs_list, ndmin=2).T 
        targets = numpy.array(targets_list, ndmin=2).T 
 
  
        # calculate signals into hidden layer 
        hidden_inputs = numpy.dot(self.weightsInputHidden, inputs) 
        # calculate the signals emerging from hidden layer 
        hidden_outputs = self.activationFunction(hidden_inputs) 
 
  
        # calculate signals into final output layer 
        final_inputs = numpy.dot(self.weightsHiddenOutput, hidden_outputs) 
        # calculate the signals emerging from final output layer 
        final_outputs = self.activationFunction(final_inputs) 

        output_errors = targets - final_outputs
        #errors​ hidden​ = weights​ Transpose​ hidden_output​ ∙ errors​ output
        hidden_errors = numpy.dot(self.weightsHiddenOutput.T, output_errors)  
        # update the weights for the links between the hidden and output layers 
        # the matrix of outputs from the previous layer, is transposed.
        # In effect this means the column of outputs becomes a row of outputs.
        self.weightsHiddenOutput += self.learningRate * numpy.dot((output_errors*final_outputs*(1-final_outputs)),numpy.transpose(hidden_outputs))
        self.weightsInputHidden += self.learningRate * numpy.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),numpy.transpose(inputs))

    def query(self,inputs_list):
        # convert inputs list to 2d array 
        inputs = numpy.array(inputs_list, ndmin=2).T
        #X​ hidden​ =​ W​ input_hidden​ ∙ I        
        hidden_inputs = numpy.dot(self.weightsInputHidden, inputs) 
        #O​ hidden​ = sigmoid( ​ X​ hidden ) 
        hidden_outputs = self.activationFunction(hidden_inputs)
        # calculate signals into final output layer 
        final_inputs = numpy.dot(self.weightsHiddenOutput, hidden_outputs) 
        # calculate the signals emerging from final output layer 
        final_outputs = self.activationFunction(final_inputs)
        return final_outputs   



inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.3

network = artificialNeuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate)
print("Before training :",network.query([1.0,0.5,-1.5]))
network.train([1.0,0.5,-1.5],[5.0,0.1,-2])
print("After training  :",network.query([1.0,0.5,-1.5]))

