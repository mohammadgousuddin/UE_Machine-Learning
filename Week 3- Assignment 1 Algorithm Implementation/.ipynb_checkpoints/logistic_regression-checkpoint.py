import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress runtime warnings (often from overflow/underflow in sigmoid)
warnings.filterwarnings("ignore")

# --- 1. Data Loading and Preprocessing ---

# Load the dataset
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Error: data.csv not found. Please make sure the file is in the same directory.")
    exit()

# Separate features (X) and target (y)
# The 'Result' column is the target
X = data.drop('Result', axis=1).values
y = data['Result'].values

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
# Standardization (Z-score normalization) is crucial for gradient-based algorithms
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Helper Functions ---

def accuracy(y_true, y_pred):
    """Calculates the accuracy percentage."""
    return np.sum(y_true == y_pred) / len(y_true)

# --- 2. Algorithm 1: Perceptron ---
# Linear classifier (update rule based)

class Perceptron:
    """
    Perceptron classifier implemented from scratch using NumPy.
    
    Parameters:
    - learning_rate (float): The step size for weight updates.
    - n_iters (int): The number of passes over the training dataset.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _activation_fn(self, x):
        """Heaviside step function."""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """Train the Perceptron model."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure y is in {0, 1}
        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_fn(linear_output)

                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """Predict class labels for samples in X."""
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_fn(linear_output)
        return y_predicted

# --- 3. Algorithm 2: Logistic Regression ---
# Probabilistic binary classifier (gradient descent)

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using NumPy.
    
    Parameters:
    - learning_rate (float): The step size for gradient descent.
    - n_iters (int): The number of passes over the training dataset.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip to avoid overflow/underflow
        z_clipped = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y):
        """Train the Logistic Regression model using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Calculate gradients (derivative of log-loss)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predict class labels for samples in X."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_probs = self._sigmoid(linear_model)
        # Convert probabilities to class labels
        y_predicted_labels = [1 if i > 0.5 else 0 for i in y_predicted_probs]
        return np.array(y_predicted_labels)

# --- 4. Algorithm 3: Simple Neural Network (1 Hidden Layer) ---
# Using backpropagation

class SimpleNeuralNetwork:
    """
    A simple 1-hidden-layer neural network implemented from scratch.
    
    Parameters:
    - input_nodes (int): Number of features.
    - hidden_nodes (int): Number of nodes in the hidden layer.
    - output_nodes (int): Number of output nodes (1 for binary classification).
    - learning_rate (float): Step size for gradient descent.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.01):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate

        # Initialize weights with small random values
        self.weights_ih = np.random.randn(self.input_nodes, self.hidden_nodes) * 0.01
        self.bias_h = np.zeros((1, self.hidden_nodes))
        self.weights_ho = np.random.randn(self.hidden_nodes, self.output_nodes) * 0.01
        self.bias_o = np.zeros((1, self.output_nodes))

    def _sigmoid(self, z):
        z_clipped = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z_clipped))

    def _sigmoid_derivative(self, s):
        """Derivative of the sigmoid function."""
        return s * (1 - s)

    def fit(self, X, y, n_iters=1000):
        """Train the neural network using backpropagation."""
        # Reshape y to be a column vector
        y = y.reshape(-1, 1)

        for _ in range(n_iters):
            # --- Feedforward ---
            # Hidden layer
            hidden_layer_input = np.dot(X, self.weights_ih) + self.bias_h
            hidden_layer_output = self._sigmoid(hidden_layer_input)
            
            # Output layer
            output_layer_input = np.dot(hidden_layer_output, self.weights_ho) + self.bias_o
            predicted_output = self._sigmoid(output_layer_input)

            # --- Backpropagation ---
            
            # Calculate output layer error
            output_error = y - predicted_output
            
            # Get gradient for output layer weights
            d_predicted_output = output_error * self._sigmoid_derivative(predicted_output)
            
            # Calculate hidden layer error
            hidden_layer_error = np.dot(d_predicted_output, self.weights_ho.T)
            
            # Get gradient for hidden layer weights
            d_hidden_layer = hidden_layer_error * self._sigmoid_derivative(hidden_layer_output)

            # --- Update Weights and Biases ---
            self.weights_ho += np.dot(hidden_layer_output.T, d_predicted_output) * self.lr
            self.bias_o += np.sum(d_predicted_output, axis=0, keepdims=True) * self.lr
            self.weights_ih += np.dot(X.T, d_hidden_layer) * self.lr
            self.bias_h += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.lr

    def predict(self, X):
        """Predict class labels for samples in X."""
        # Feedforward
        hidden_layer_input = np.dot(X, self.weights_ih) + self.bias_h
        hidden_layer_output = self._sigmoid(hidden_layer_input)
        
        output_layer_input = np.dot(hidden_layer_output, self.weights_ho) + self.bias_o
        predicted_output_probs = self._sigmoid(output_layer_input)
        
        # Convert probabilities to class labels
        predicted_labels = [1 if i > 0.5 else 0 for i in predicted_output_probs]
        return np.array(predicted_labels)

# --- 5. Training and Evaluation ---

print("Training and evaluating models...")

# Define common parameters
n_iterations = 1000
learning_rt = 0.01

# Perceptron
p = Perceptron(learning_rate=learning_rt, n_iters=n_iterations)
p.fit(X_train, y_train)
p_preds = p.predict(X_test)
p_accuracy = accuracy(y_test, p_preds)
print(f"Perceptron Test Accuracy: {p_accuracy*100:.2f}%")

# Logistic Regression
lr = LogisticRegression(learning_rate=learning_rt, n_iters=n_iterations)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_accuracy = accuracy(y_test, lr_preds)
print(f"Logistic Regression Test Accuracy: {lr_accuracy*100:.2f}%")

# Simple Neural Network
n_features = X_train.shape[1]
hidden_nodes = 10 # You can tune this parameter
nn = SimpleNeuralNetwork(input_nodes=n_features, hidden_nodes=hidden_nodes, output_nodes=1, learning_rate=learning_rt)
nn.fit(X_train, y_train, n_iters=n_iterations)
nn_preds = nn.predict(X_test)
nn_accuracy = accuracy(y_test, nn_preds)
print(f"Simple Neural Network (1 Hidden Layer) Test Accuracy: {nn_accuracy*100:.2f}%")