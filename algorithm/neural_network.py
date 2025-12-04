from __future__ import annotations

from numpy.typing import NDArray
from algorithm.activation_function import ActivationFunction
from algorithm.genome import Genome


class NeuralNetwork:

    """
    Represents a feed-forward neural network.
    The input, output, and hidden layers are all dense layers.

    Attributes
    ----------
    layers : list[DenseLayer]
        A list containing the hidden layers and the output layer.

    Methods
    -------
    from_genome(genome: Genome) -> NeuralNetwork (static)
        Creates a neural network from a genome.
    forward(genome: Genome) -> NDArray[float]
        Executes a full forward pass of the neural network.

    Notes
    -----
    There is no explicit input layer. The input size is implicitly defined
    by the shape of the first layer's weight matrix.

    Backpropagation is not done, as the neural network is meant to be
    used for genetic algorithms.
    """

    def __init__(self, layers: list[DenseLayer]) -> None:

        self.layers: list[DenseLayer] = layers

    @staticmethod
    def from_genome(genome: Genome) -> NeuralNetwork:

        """
        Creates a neural network from a genome.

        Parameters
        ----------
        genome : Genome
            The genome to create the neural network from.

        Returns
        -------
        NeuralNetwork
            The neural network created from the genome.

        """

        # Gets the weights and biases of the genome.
        layer_weights: list[tuple[NDArray[float], NDArray[float]]] = genome.get_layer_weights()

        # Computes a list with the size of each layer (excluding input).
        sizes: list[int] = genome.topology + [genome.output_size]

        # Creates each layer of the neural network.
        layers: list[DenseLayer] = [
            DenseLayer(size, act, W=W, b=b)
            for size, act, (W, b) in zip(sizes, genome.activations, layer_weights)
        ]

        return NeuralNetwork(layers)

    def forward(self, X: NDArray[float]) -> NDArray[float]:

        """
        Executes a full forward pass of the neural network.

        Parameters
        ----------
        X : NDArray[float]
            The input data passed into the neural network.

        Returns
        -------
        NDArray[float]
            The neural network's output data.
        """

        x: NDArray[float] = X.copy()

        # Runs the forward pass of each layer.
        for layer in self.layers:
            x = layer.forward(x)

        return x


class DenseLayer:

    """
    Represents a dense layer in a neural network.
    Dense layers are fully connected to each other.

    Attributes
    ----------
    size : int
        The number of neurons in the layer.
    activation: ActivationFunction
        An activation function instance to use in the layer.
    W : NDArray[float]
        An array containing the weights of the layer's neurons.
    b : NDArray[float]
        An array containing the biases of the layer's neurons.

    Methods
    -------
    forward(X: NDArray[float]) -> NDArray[float]
        Executes a forward pass of the layer.
    """

    def __init__(self, size: int, activation: ActivationFunction, W: NDArray[float], b: NDArray[float]):

        self.size: int = size
        self.activation: ActivationFunction = activation
        self.W: NDArray[float] = W
        self.b: NDArray[float] = b

    def forward(self, x: NDArray[float]) -> NDArray[float]:

        """
        Executes a forward pass of the layer.
        Data goes through the layer's activation function and the result
        is then output.

        Parameters
        ----------
        x : NDArray[float]
            An array containing the data received by the layer.

        Returns
        -------
        NDArray[float]
            An array containing the output data of the layer.
        """

        # Computes the weighted sum of inputs + bias for each neuron.
        # First, the weight matrix is transposed so it can be multiplied by the data matrix.
        # Transposing ensures the inner dimensions match (needed for matrix multiplication).
        # Finally, each neuron's bias is added.
        # This is essentially y=ax+b for each neuron.
        z = x @ self.W.T + self.b

        # Runs the data through the layer's activation function.
        return self.activation.forward(z)
