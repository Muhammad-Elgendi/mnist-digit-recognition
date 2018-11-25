import numpy
import matplotlib.pyplot

trainFile =open('/media/muhammad/disk/MachineLearning/mnist_train_100.csv','r')
trainingExamples = trainFile.readlines()
trainFile.close()
raw_values = trainingExamples[0].split(',')
# print(raw_values)
image_array = numpy.asfarray(raw_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation=None )
matplotlib.pyplot.show()
# matplotlib.pyplot.savefig('/media/muhammad/disk/MachineLearning/myfig.png')

scaled_inputs = (numpy.asfarray(raw_values[1:]) / 255 *0.99) +0.01
print(scaled_inputs)
output_nodes = 10
targets = numpy.zeros(output_nodes) + 0.01
targets[int(raw_values[0])] =0.99
print(targets)