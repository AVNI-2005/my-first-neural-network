print ("hello")
import matplotlib as plt 
import math
import numpy as np
import nnfs
from nnfs.datasets import vertical_data


nnfs.init()        #np.random.seed(0)
# X = [[1 , 2 , 3 , 2.5 ],
#     [2.0 , 5.0 , -1.0 ,2.0 ],
#     [-1.5 , 2.7 , 3.3 , -0.8]]

# X , y = spiral_data( 9,2)

class layer_dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.1* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

class loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class loss_CategoraicalCrossentropy(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred , 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)

        negative_log_likelihods = -np.log(correct_confidences)
        return negative_log_likelihods



#X, y = spiral_data(samples = 100 , classes = 3)
X, y = vertical_data(samples = 100 , classes = 3)

dense1 = layer_dense(2, 3)
activation1 = Activation_ReLu()

dense2 = layer_dense(3, 3)
activation2 = activation_softmax()


loss_function = loss_CategoraicalCrossentropy()

lowest_loss = 99999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
final_result = []
final_iter = []

for iterations in range(1000000):
    dense1.weights += 0.5 * np.random.randn(2,3)
    dense1.biases += 0.5 * np.random.randn(1,3)
    dense2.weights += 0.5 * np.random.randn(3,3)
    dense2.biases += 0.5 * np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis = 1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss :
        print('New set of weights found, iteration:',iterations,'loss:', loss, 'acc:', accuracy)
        best_dense1_biases = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        final_result.append(lowest_loss)
        final_iter.append(iterations)

plt.plot(final_iter, final_result)
plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.title("neural network's loss regression")
plt.show()