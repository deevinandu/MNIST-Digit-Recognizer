import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare the data
data = pd.read_csv(r"/content/mnist_test.csv")
data = np.array(data)
np.random.shuffle(data)

# Split the dataset 
split_index = int(0.8 * data.shape[0])
data_train = data[:split_index].T
data_dev = data[split_index:].T

Y_train = data_train[0].astype(int)
X_train = data_train[1:] / 255.
Y_dev = data_dev[0].astype(int)
X_dev = data_dev[1:] / 255.

# Hyperparameters
hidden_neurons = 64
input_neurons = 784
output_neurons = 10

def init_params():
    W1 = np.random.randn(hidden_neurons, input_neurons) * np.sqrt(2./input_neurons)
    b1 = np.zeros((hidden_neurons, 1))
    W2 = np.random.randn(output_neurons, hidden_neurons) * np.sqrt(2./hidden_neurons)
    b2 = np.zeros((output_neurons, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / exp_Z.sum(axis=0, keepdims=True)

def one_hot(Y):
    Y = Y.astype(int)
    one_hot_Y = np.zeros((output_neurons, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

# Forward and backward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T @ dZ2 * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Gradient descent
def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

# Prediction and accuracy
def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def compute_loss(A2, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    return -np.sum(one_hot_Y * np.log(A2 + 1e-9)) / m

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)
        if i % 10 == 0 or i == iterations - 1:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            loss = compute_loss(A2, Y)
            print(f"Iteration {i}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.01, iterations=1000)

# Generates predictions using the input data and the trained parameters
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Tests a single prediction using the trained parameters on a specific index
def test_function(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]  # Shape: (784, 1)
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = int(Y_train[index])
    print("Prediction:", int(prediction[0]))
    print("Actual Label:", label)
    # Display the image
    image_display = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image_display, interpolation='nearest')
    plt.title(f"Prediction: {int(prediction[0])}, Label: {label}")
    plt.axis('off')
    plt.savefig('test_image.png')

test_function(5, W1, b1, W2, b2)
test_function(133, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_accuracy(dev_predictions, Y_dev)
print("Development Set Accuracy:", accuracy)

# new test data
data_new = pd.read_csv(r"/content/mnist_test.csv")

def preprocess_features(data):
    X = data.drop(columns=['label']).values
    X = X / 255
    X = X.reshape(X.shape[0], -1)
    return X

def preprocess_labels(data):
    Y = data['label'].values
    return Y

X_new = preprocess_features(data_new)
Y_new = preprocess_labels(data_new)

# Run new data through the neural network
_, _, _, A2_new = forward_prop(W1, b1, W2, b2, X_new.T)
predictions_new = get_predictions(A2_new)
accuracy_new = get_accuracy(predictions_new, Y_new)
print("Accuracy on New Dataset:", accuracy_new)