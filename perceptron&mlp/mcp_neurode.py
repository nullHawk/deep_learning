# This is implementation of McCulloch and Pitt's Neurode model

class neurode:
    def __init__(self, *X, theta,):
        self.theta = theta
        self.X = X

    def _activation(self):
        """
        Activation function for the neurode.
        Returns 1 if the sum of inputs exceeds the threshold, otherwise returns 0.
        """
        input_sum = sum(x for x in self.X)
        return 1 if input_sum >= self.theta else 0
    
    def process(self):
        """
        Process the inputs through the neurode and return the output.
        """
        return self._activation()

if __name__ == "__main__":
    # Example usage
    neurode_instance = neurode(1, 1, 1, theta=0)
    output = neurode_instance.process()
    print(f"Output of the neurode: {output}")  # Should print 1 since 1+1+1 >= 1