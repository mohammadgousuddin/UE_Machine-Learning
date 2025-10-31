
import numpy as np

class Perceptron:
    """
    A simple implementation of the Perceptron algorithm.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initializes the Perceptron classifier.

        Args:
            learning_rate (float): The learning rate for weight updates.
            n_iters (int): The number of passes over the training dataset.
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _activation(self, x):
        """
        The activation function. In the case of Perceptron, it's a simple step function.
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Trains the Perceptron model.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert y to a numpy array if it's not already
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_output)

                # Perceptron update rule
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predicts the class labels for the given input samples.

        Args:
            X (np.ndarray): The input samples to predict.

        Returns:
            np.ndarray: The predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation(linear_output)
        return y_predicted

if __name__ == '__main__':
    # Example usage of the Perceptron classifier
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()
