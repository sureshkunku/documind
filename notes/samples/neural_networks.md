# Neural Networks

## What is a neural network
A neural network is a model made of layers of connected neurons. Each connection has a weight. The input layer receives data, hidden layers transform it, and the output layer gives the prediction.

## Activation functions
An activation function adds non-linearity, so the network can learn complex patterns. ReLU returns max(0, x) and is the default choice for hidden layers. Sigmoid squashes values into the range 0 to 1 and is used for binary classification outputs.

## Backpropagation
Backpropagation computes the gradient of the loss with respect to every weight, using the chain rule from the output layer back to the input layer. The gradients tell each weight which direction to move to reduce the loss.

## Gradient descent
Gradient descent updates each weight by subtracting the learning rate multiplied by its gradient. A learning rate that is too high makes training diverge; too low makes it very slow. A common starting value is 0.001.

## Overfitting
Overfitting happens when a model memorises the training data and performs badly on new data. Dropout, early stopping and more training data all reduce overfitting.