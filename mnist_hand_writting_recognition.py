import numpy
import scipy.special
import scipy.ndimage
import matplotlib.pyplot
import imageio

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

def loadDataList(dir):
    myfile =open(dir,'r')
    records = myfile.readlines()
    myfile.close()
    return records



inputNodes = 28*28
hiddenNodes = 200
outputNodes = 10
learningRate = 0.1

# epochs is the number of times the training data set is used for training (the more higher the more overfitting)
epochs = 5

network = artificialNeuralNetwork(inputNodes,hiddenNodes,outputNodes,learningRate)

# load the mnist training data CSV file into a list
train_set = loadDataList('/media/muhammad/disk/MachineLearning/mnist_train.csv')

# preparing the training Examples and the target output and training our ANN

for epoch in range(epochs):
    # go through all records in the training data set
    for record in train_set:

        # split the record by the ',' commas
        raw_values = record.split(',')
        # scale and shift the inputs to be appropriate to logistic (sigmoid) function 
        scaled_inputs = (numpy.asfarray(raw_values[1:]) / 255 *0.99) +0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(outputNodes) + 0.01
        # raw_values[0] is the target label for this record
        targets[(raw_values[0])] =0.99

        # e.g for 3 --> 0.01 0.01 0.01 0.99 0.01 0.01  0.01 0.01  0.01 0.01

        network.train(scaled_inputs, targets)

        # create rotated variations
        # rotated anticlockwise by x degrees
        #cval : scalar, optional
            #Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
        #order : int, optional
            #The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
        #reshape : bool, optional
            #If reshape is true, the output shape is adapted so that the input array is contained completely in the output. Default is True.
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(scaled_inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        network.train(inputs_plusx_img.reshape(784), targets)

        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(scaled_inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        network.train(inputs_minusx_img.reshape(784), targets)

# load the mnist test data CSV file into a list
test_set = loadDataList('/media/muhammad/disk/MachineLearning/mnist_test.csv')

# query the neural network with test data

# performance for how well the network performs, initially empty
performance = []

# go through all the records in the test data set
for record in test_set:

    # split the record by the ',' commas
    raw_values = record.split(',')
    # correct answer is first value
    correct_label = int(raw_values[0])
    # scale and shift the inputs to be appropriate to logistic (sigmoid) function 
    scaled_inputs = (numpy.asfarray(raw_values[1:]) / 255 *0.99) +0.01
    # query the network
    outputs = network.query(scaled_inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        performance.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        performance.append(0)


# calculate the performance score, the fraction of correct answers
performance_array = numpy.asarray(performance)
print ("Accuracy = ", performance_array.sum() / performance_array.size)

# test the neural network with my own images

# load image data from png files into an array
print ("loading ... 2828_my_own_6.png")
img_array = imageio.imread('2828_my_own_6.png', as_gray=True)
    
# reshape from 28x28 to list of 784 values, invert values
img_data  = 255.0 - img_array.reshape(784)
    
# then scale data to range from 0.01 to 1.0
img_data = (img_data / 255.0 * 0.99) + 0.01

# query the network
outputs = network.query(img_data)
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network output ", label)