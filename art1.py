import jax
import jax.numpy as jnp
from jax import jit


class ART1:
    def __init__(self, num_input, vigilance):
        self.num_input = num_input
        self.vigilance = vigilance
        self.W = jnp.ones((1, num_input))  # Initial weights
        self.T = 1.0  # Threshold
        self.alpha = 1.0  # Choice parameter
        self.beta = 0.1  # Learning rate

    def reset(self):
        self.W = jnp.ones((1, self.num_input))

    def compute_f1(self, input_pattern):
        return input_pattern / (self.alpha + jnp.sum(input_pattern))

    @jit
    def train(self, input_pattern):
        f1 = self.compute_f1(input_pattern)
        while True:
            net_input = jnp.dot(self.W, f1)
            choice = jnp.argmax(net_input)
            net_input_max = jnp.max(net_input)
            if net_input_max >= self.T:
                if jnp.all(f1 <= self.W[choice]):
                    self.W = jax.ops.index_update(self.W, jax.ops.index[choice],
                                                  self.beta * f1 + (1 - self.beta) * self.W[choice])
                    break
            else:
                self.W = jnp.vstack((self.W, input_pattern))
                break

    @jit
    def predict(self, input_pattern):
        f1 = self.compute_f1(input_pattern)
        net_input = jnp.dot(self.W, f1)
        choice = jnp.argmax(net_input)
        return choice


# Example usage:
if __name__ == "__main__":
    num_input = 5  # Number of input nodes
    vigilance = 0.7  # Vigilance parameter
    art_network = ART1(num_input, vigilance)

    # Example training data
    training_data = jnp.array([[1, 0, 0, 1, 1],
                               [0, 1, 0, 1, 0],
                               [1, 1, 0, 1, 0]])

    # Training the ART1 network
    for pattern in training_data:
        art_network.train(pattern)

    # Example test data
    test_pattern = jnp.array([1, 0, 0, 1, 0])

    # Predicting the category of the test pattern
    category = art_network.predict(test_pattern)
    print("Predicted category:", category)
