import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    A simple Neural Network with one hidden layer using backpropagation.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, n_iters=1000):
        """
        Initializes the Neural Network.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output neurons (classes).
            learning_rate (float): The learning rate for weight updates.
            n_iters (int): The number of iterations for training.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def _sigmoid(self, x):
        """
        The sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid activation function.
        """
        return x * (1 - x)

    def fit(self, X, y):
        """
        Trains the Neural Network using backpropagation.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.
        """
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            # Forward propagation
            # Hidden layer
            self.Z1 = np.dot(X, self.W1) + self.b1
            self.A1 = self._sigmoid(self.Z1)

            # Output layer
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = self._sigmoid(self.Z2)

            # Backpropagation
            # Output layer error
            dZ2 = self.A2 - y.reshape(-1, 1)  # Ensure y is column vector
            dW2 = (1 / n_samples) * np.dot(self.A1.T, dZ2)
            db2 = (1 / n_samples) * np.sum(dZ2, axis=0, keepdims=True)

            # Hidden layer error
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._sigmoid_derivative(self.A1)
            dW1 = (1 / n_samples) * np.dot(X.T, dZ1)
            db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)

            # Update weights and biases
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

    def predict(self, X):
        """
        Predicts the class labels for the given input samples.

        Args:
            X (np.ndarray): The input samples to predict.

        Returns:
            np.ndarray: The predicted class labels (0 or 1).
        """
        # Forward propagation
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._sigmoid(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)

        y_predicted_cls = np.array([1 if i > 0.5 else 0 for i in A2])
        return y_predicted_cls

if __name__ == '__main__':
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Generate a synthetic dataset for binary classification
    X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Initialize and train the Neural Network
    input_size = X_train.shape[1]
    hidden_size = 4  # Number of neurons in the hidden layer
    output_size = 1  # Binary classification

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1, n_iters=10000)
    nn.fit(X_train, y_train)
    predictions = nn.predict(X_test)

    print("Neural Network classification accuracy:", accuracy(y_test, predictions))

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Dataset")

    def plot_decision_boundary(pred_func):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and the training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.show()

    plot_decision_boundary(lambda x: nn.predict(x))