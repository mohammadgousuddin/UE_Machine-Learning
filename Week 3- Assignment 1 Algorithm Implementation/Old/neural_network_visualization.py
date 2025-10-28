import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, n_iters=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        n_samples = X.shape[0]

        for _ in range(self.n_iters):
            self.Z1 = np.dot(X, self.W1) + self.b1
            self.A1 = self._sigmoid(self.Z1)

            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = self._sigmoid(self.Z2)

            dZ2 = self.A2 - y.reshape(-1, 1)
            dW2 = (1 / n_samples) * np.dot(self.A1.T, dZ2)
            db2 = (1 / n_samples) * np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self._sigmoid_derivative(self.A1)
            dW1 = (1 / n_samples) * np.dot(X.T, dZ1)
            db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._sigmoid(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._sigmoid(Z2)

        y_predicted_cls = np.array([1 if i > 0.5 else 0 for i in A2])
        return y_predicted_cls

if __name__ == '__main__':
    X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    input_size = X_train.shape[1]
    hidden_size = 4
    output_size = 1

    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1, n_iters=10000)
    nn.fit(X_train, y_train)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title("Dataset")

    def plot_decision_boundary(pred_func):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.title("Neural Network Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    plot_decision_boundary(lambda x: nn.predict(x))