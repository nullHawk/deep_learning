import numpy
import math

def sigmoid(x):
    return 1/(1+math.e ** (-x))

if __name__ == "__main__":
    # Example usage
    x = numpy.array([0, 1, 2, 3, 4, 5])
    sigmoid_values = sigmoid(x)
    print(f"Sigmoid values: {sigmoid_values}")  # Should print sigmoid values for each element in x