import numpy as np
import random

"""
INPUT                    HIDDEN                    OUTPUT

 X        W^(1)          Z^(2)         W^(2)       Z^(3)
                         a^(2)                     a^(3) = y' (our prediction)


X - Matrix of inputs (our feature vector)

W^(n) - Weights for layer n.

Z^(n) - weights * input + bias for layer n. for Z^(2) = X W^(1) + b^(1) , Z^(3) = a^(2) W^(2) + b^(2)

a^(n) - Activation of layer n.  f( Z^(n) ) where f is the activation func.
"""
class NeuralNetwork:

    def __init__(self, sizes):
        """
        Size is an array where its dim is the depth of the network and each
        element defines the size of that layer. For this simple example want
        [2, 3, 1] (2 inputs 3 in hidden layer and 1 output)
        """
        self.sizes = sizes

        self.num_layers = len(sizes)

        """
        biases is an array of random numbers from a normal distibution.
        Each entry is a column vector with a size equal to the number
        of neurons in that layer.

        So for a hidden layer of 3 and output layer of 1:

        [ | b_11 | , | b_21 | ]
          | b_12 |
          | b_13 |

        """
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        """
        Weights is an array containing matrices for each connecting layer.
        The dimensions of this matrix is equal to the number of connections.

        So for a input 2   hidden 3   output 1

        there are 6 connections between the input and the hidden layer
        and 3 connections between the hidden layer and output layer.

        [
            [
                [w^(1)_11, w^(1)_21],
                [w^(1)_12, w^(1)_22],
                [w^(1)_13, w^(1)_23]

            ],  (dim = # neurons in hidden layer, # inputs )

            [
                [w^(2)_11, w^(2)_21, w^(2)_31],
            ]
        ]
        """
        self.weights = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]

        self.activations = []
        self.Z = []
        self.errors = []

    def forward_prop(self, x):
        """
        This pushes forward in our network without any gradient decent or
        back prop.

        For layer n we are doing ( given x = a^(n), w = W^(n), b = b^(n) )

        example input layer x (dim=2) to hidden layer W^(1) (dim=3):

        W^(1) * x = |  [w^(1)_11, w^(1)_21] | o | x_11 |
                    |  [w^(1)_12, w^(1)_22] |   | x_12 |
                    |  [w^(1)_13, w^(1)_23] |
                            3 x 2                2 x 1

                  =   | w^(1)_11 * x_11 + w^(1)_21 * x_12 |
                      | w^(1)_12 * x_11 + w^(1)_22 * x_12 |
                      | w^(1)_13 * x_11 + w^(1)_23 * x_12 |
                                     3 x 1


        b^(1) = | b^(1)_1 |
                | b^(1)_2 |
                | b^(1)_3 |

        Z^(1) = x*W^(1) + b^(1)

              = | w^(1)_11 * x_11 + w^(1)_21 * x_12 |   | b^(1)_1 |
                | w^(1)_12 * x_11 + w^(1)_22 * x_12 | + | b^(1)_2 |
                | w^(1)_13 * x_11 + w^(1)_23 * x_12 |   | b^(1)_3 |

              = | (w^(1)_11 * x_11 + w^(1)_21 * x_12) + b^(1)_1 |
                | (w^(1)_12 * x_11 + w^(1)_22 * x_12) + b^(1)_2 |
                | (w^(1)_13 * x_11 + w^(1)_23 * x_12) + b^(1)_3 |

              = | z^(1)_1 |
                | z^(1)_2 |
                | z^(1)_3 |

        f(z) = 1 / (1 + exp(-z))

        A^(1) = f(Z^(1))

              = | 1 / {1 + exp(-z^(1)_1)} |
                | 1 / {1 + exp(-z^(1)_2)} |
                | 1 / {1 + exp(-z^(1)_3)} |

              = | a^(1)_1 |
                | a^(1)_2 |
                | a^(1)_3 |
        """
        self.activations = [x]
        activation = x
        self.Z = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            self.Z.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)

    def back_prop(self, X, y):
        # the gradient of the cost wrt. biases
        grad_b = [np.zeros(b.shape) for b in self.biases]
        # the gradient of the cost wrt. weights
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # Step one: forward propagation
        self.forward_prop(X)

        # Step two: calcuate the error in the final layer delta^L
        self.errors = [None] * len(self.weights)
        #                 Hadamard (element-wise) Product
        #                         ^
        # Delta_L = (y^(hat) - y) o sig_prime( Z^L )
        delta_L = (
            self.cost_derivative(self.activations[-1], y) *
            self.sigmoid_prime(self.Z[-1])
        )
        self.errors[-1] = delta_L

        # Step three: calculate the error for every other layer
        for l in reversed(range(self.num_layers - 2)):
            #                 Hadamard (element-wise) Product
            #                         ^
            # Delta = W^T * D^(l + 1) o sig_prime( Z^l )
            delta = (
                np.dot(self.weights[l + 1].T, self.errors[l+1]) *
                self.sigmoid_prime(self.Z[l])
            )
            self.errors[l] = delta

        # Step 4: calcuate the gradient of C w.r.t b and w.
        for l in reversed(range(1, self.num_layers)):
            # grad_w = a^(l - 1) delta^l
            # but because of dimensions we want:
            # delta^l (a^(l - 1))^T
            # Also note: dim{error} = n - 1, dim{activations} = n
            grad_w[l-1] = np.dot(self.errors[l-1], self.activations[l - 1].T)

        grad_b = self.errors

        return grad_b, grad_w

    def batch_update(self, batch, learning_rate):
        """
        Updates the weights and biases via grad desc. by applying back
        prop to a small batch.
        """
        # the gradient of the cost wrt. biases
        grad_b = [np.zeros(b.shape) for b in self.biases]
        # the gradient of the cost wrt. weights
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_grad_b, delta_grad_w = self.back_prop(x, y)
            grad_b = [nb+dnb for nb, dnb in zip(grad_b, delta_grad_b)]
            grad_w = [nw+dnw for nw, dnw in zip(grad_w, delta_grad_w)]
        self.weights = [w-(learning_rate/len(batch))*nw
                        for w, nw in zip(self.weights, grad_w)]
        self.biases = [b-(learning_rate/len(batch))*nb
                       for b, nb in zip(self.biases, grad_b)]

    def SGD(self, training_data, epochs, batch_size, learning_rate,
            test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent. 

        When `test_data` is passed, the accuracy of the NN will be evaluated 
        after each epoch.
        """
        training_cost = []
        training_accuracy = []
        test_cost = []
        test_accuracy = []
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            batches = [
                training_data[k:k+batch_size]
                for k in range(0, n, batch_size)]
            for batch in batches:
                self.batch_update(batch, learning_rate)
            training_cost.append(
                self.total_cost(training_data)
            )
            training_accuracy.append(
                self.evaluate(training_data, convert=True)
            )
            if test_data:
                test_cost.append(
                    self.total_cost(test_data, convert=True)
                )
                test_accuracy.append(
                    self.evaluate(test_data)
                )
                print("Epoch {0}: {1} / {2} ({3})".format(
                        j,
                        self.evaluate(test_data),
                        n_test,
                        self.evaluate(test_data) / n_test
                    )
                )
            else:
                print("Epoch {0}: {1} / {2} ({3})".format(
                        j,
                        self.evaluate(training_data, convert=True),
                        len(training_data),
                        self.evaluate(training_data, convert=True) / len(training_data)
                    )
                )

        return training_cost, training_accuracy, test_cost, test_accuracy

    def feedforward(self, a):
        """
            Feed forward without saving `self.Z` or `self.activations`
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)

        return a

    def evaluate(self, test_data, convert=False):
        """
        Return the number of test inputs with the correct prediction
        """
        if convert:
            test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                            for (x, y) in test_data]
        else:
            test_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def predict(self, x):
        return self.feedforward(x)

    def total_cost(self, data, convert=False):
        tot = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                vec = np.zeros((10, 1))
                vec[y] = 1.0
                y = vec
            tot += self.cost(a, y) / len(data)
        return tot

    def cost(self, a, y):
        return 0.5 * np.linalg.norm(a-y)**2

    def cost_derivative(self, a, y):
        """
        If using a quadratic cost function:

        y = training output
        a^(n) = yHat = output (activations / predictions)

        C = 1/2 sum_j (y_j - a^(n)_j)^2

        Want to compute del_a C (derv of the cost w.r.t. the activations) which
        is a vector containing elements:

         d C
        ------     = a^(n) - y
         d a^(n)_j
        """
        return (a - y)

    def sigmoid(self, z):
        """
        Our activation function
        """
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """
        Derivative of the sigmoid function
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))
