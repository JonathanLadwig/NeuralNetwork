import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
from matplotlib import colors
from tkinter import *

# input data
inputs = np.array([[0, 1, 0], #Green
                   [0, 1, 1], #Cyan
                   [0, 0, 0], #Black
                   [1, 0, 0], #Red
                   [1, 1, 1], #White
                   [1, 1, 0], #Yellow
                   [1, 0, 1], #Pink
                   [0, 0, 1]]) #Blue
# output data 0 is black, 1 is white
outputs = np.array([[0], [0], [1], [1], [0], [0], [0], [1]])

#creating the network class
class NeuralNetwork:

    #initialising the variables
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        #initialise weights
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    # activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    #forward propagation to produce an output
    def forwardpropagation(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    #going backwards through the network and updating each of the weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    #train the neural network (epoch) many times
    def train(self, epochs=1000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.forwardpropagation()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create neural network
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

#colours to rgb
val = ""
while val != "End":
    val = input("Enter colour hexcode or name: ")
    if val == "End":
        break
    RGB = colors.to_rgb(val)
    predictedOutput = NN.predict(RGB)
    print(predictedOutput)
    if predictedOutput >= 0.25:
        print("White")
    else:
        print("Black")

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

root = Tk()
myLabel = Label(root, text = "Hi there")
myLabel.pack()
root.mainloop()