import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = np.array([1 if i > 0.5 else 0 for i in y_predicted])
        return y_predicted_cls

if __name__ == '__main__':
    X_plot, y_plot = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train_plot, X_test_plot, y_train_plot, y_test_plot = train_test_split(
        X_plot, y_plot, test_size=0.2, random_state=123
    )
    regressor_plot = LogisticRegression(learning_rate=0.01, n_iters=1000)
    regressor_plot.fit(X_train_plot, y_train_plot)

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

    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()