import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            for x_i, y_i in zip(X, y):
                prediction = self.predict(x_i)
                if prediction != y_i:
                    self.weights += self.learning_rate * y_i * x_i
                    self.bias += self.learning_rate * y_i

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self._activate_fn(linear_output)

    def _activate_fn(self, x):
        return 1 if x >= 0 else -1

# Example usage
if __name__ == '__main__':
    learning_rate = 0.1
    epoch = 10

    # Sample binary classification data (linearly separable)
    # We are using a simple 1D dataset: x and y ∈ {-1, +1}
    train_X = np.array([[-2], [-1], [1], [2]])     # shape (4, 1)
    train_y = np.array([-1, -1, 1, 1])             # labels

    p = Perceptron(learning_rate=learning_rate, epochs=epoch)
    p.fit(train_X, train_y)

    # Test predictions
    test_X = np.array([[-1.5], [0], [1.5]])
    for x in test_X:
        print(f"Input: {x}, Prediction: {p.predict(x)}")
