import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    A simple implementation of Logistic Regression using gradient descent.
    """

    def __init__(self, learning_rate=0.001, n_iters=1000):
        """
        Initializes the Logistic Regression classifier.

        Args:
            learning_rate (float): The learning rate for weight updates.
            n_iters (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        """
        The sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """
        Trains the Logistic Regression model using gradient descent.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts the class labels for the given input samples.

        Args:
            X (np.ndarray): The input samples to predict.

        Returns:
            np.ndarray: The predicted class labels (0 or 1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = np.array([1 if i > 0.5 else 0 for i in y_predicted])
        return y_predicted_cls

if __name__ == '__main__':
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X_plot, y_plot = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
        X_plot, y_plot, test_size=0.2, random_state=123
    )
    regressor_plot = LogisticRegression(learning_rate=0.01, n_iters=1000)
    regressor_plot.fit(X_train_plot, y_train_plot)
    predictions_plot = regressor_plot.predict(X_test_plot)

    print("Logistic Regression classification accuracy:", accuracy(y_test_plot, predictions_plot))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train_plot[:, 0], X_train_plot[:, 1], marker="o", c=y_train_plot)

    x0_1 = np.amin(X_train_plot[:, 0])
    x0_2 = np.amax(X_train_plot[:, 0])
    x1_1 = (-regressor_plot.weights[0] * x0_1 - regressor_plot.bias) / regressor_plot.weights[1]
    x1_2 = (-regressor_plot.weights[0] * x0_2 - regressor_plot.bias) / regressor_plot.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train_plot[:, 1])
    ymax = np.amax(X_train_plot[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    # plt.show()