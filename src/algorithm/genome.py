from __future__ import annotations

import numpy as np

from enum import Enum
from numpy.typing import NDArray
from config import RNG, GENOME, MUTATION
from .activation_function import ActivationFunction, ReLU, Sigmoid, Tanh


class Genome:

    """
    Represents a neural network genome for use in genetic algorithms.

    Attributes
    ----------
    input_size : int
        The number of neurons in the input layer.
    output_size : int
        The number of neurons in the output layer.
    topology : list[int]
        A list with the number of neurons in each hidden layer.
    activations : list[ActivationFunction]
        A list with the activation functions for each layer.
    weights : NDArray[float]
        A flattened array containing all weights and biases for the network.

    Methods
    -------
    random(input_size: int, output_size: int) -> Genome (static)
        Creates a genome with a random topology, activation functions, and weights.
    get_layer_weights() -> list[tuple[NDArray[float], NDArray[float]]]
        Returns the weights and biases of each layer of the genome.
    mutate()
        Applies mutations to weights, activations, and/or topology.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        topology: list[int],
        activations: list[ActivationFunction],
        weights: NDArray[float]
    ) -> None:

        self.input_size = input_size
        self.output_size = output_size
        self.topology: list[int] = topology
        self.activations: list[ActivationFunction] = activations
        self.weights: NDArray[float] = weights

    @staticmethod
    def random(input_size: int, output_size: int) -> Genome:

        """
        Creates a genome with a random topology, activation functions, and weights.

        Parameters
        ----------
        input_size : int
            The number of neurons in the input layer.
        output_size : int
            The number of neurons in the output layer.

        Returns
        -------
        Genome
            The randomly generated genome.

        Notes
        -----
        The number of hidden layers is derived from ``GENOME_MIN_LAYERS``
        and ``GENOME_MAX_LAYERS``.The number of neurons in each hidden layer is
        derived from ``GENOME_MIN_NEURONS`` and ``GENOME_MAX_NEURONS``.

        The output layer's activation function is always sigmoid.
        """

        # Computes the topology (number and sizes of hidden layers).
        num_hidden_layers: int = int(RNG.integers(GENOME.MIN_LAYERS, GENOME.MAX_LAYERS, endpoint=True))
        topology: list[int] = RNG.integers(
            GENOME.MIN_NEURONS, GENOME.MAX_NEURONS, endpoint=True, size=num_hidden_layers
        ).tolist()

        # Computes random activation functions and weights/biases for each layer.
        activations: list[ActivationFunction] = Genome._random_activations(num_hidden_layers)
        weights: NDArray[float] = Genome._random_weights(input_size, topology, output_size)

        return Genome(input_size, output_size, topology, activations, weights)

    @staticmethod
    def _random_activation() -> ActivationFunction:

        """
        Randomly selects an activation function from ReLU, Sigmoid, and Tanh.

        Returns
        -------
        ActivationFunction
            An instance of the randomly chosen activation function class.
        """

        # Selects a random activation function.
        possible_activations: list[type[ActivationFunction]] = [ReLU, Sigmoid, Tanh]
        return RNG.choice(possible_activations)()

    @staticmethod
    def _random_activations(num_layers: int) -> list[ActivationFunction]:

        """
        Randomly selects an activation function from ReLU, Sigmoid, and Tanh
        for each hidden layer of the genome.

        Parameters
        ----------
        num_layers: int
            The number of hidden layers of the genome.

        Returns
        -------
        list[ActivationFunction]
            A list of randomly chosen activation function instances.

        Notes
        -----
        A sigmoid activation is always appended at the end for the output layer.
        """

        # Creates a random activation function for each layer and adds a sigmoid function for the output layer.
        activations: list[ActivationFunction] = [Genome._random_activation() for _ in range(num_layers)]
        activations.append(Sigmoid())
        return activations

    @staticmethod
    def _random_weights(input_size: int, topology: list[int], output_size: int) -> NDArray[float]:

        """
        Creates a normal distribution of random weights for each layer of the genome.

        Parameters
        ----------
        input_size : int
            The number of neurons in the input layer.
        topology : list[int]
            A list containing the number of neurons in each hidden layer.
        output_size : int
            The number of neurons in the output layer.

        Returns
        -------
        NDArray[float]
            An array containing all weights and biases for the neural network.

        Notes
        -----
        The standard deviation of the normal distribution is controlled with ``GENOME_WEIGHTS_STD``.
        """

        # This will store the total size of weights and biases.
        total_size: int = 0

        # Computes a list of sizes (input + hidden layers + output).
        sizes: list[int] = [input_size] + topology + [output_size]

        for i in range(len(sizes) - 1):

            # The total size is the sum of two components:
            # (Size of next layer * size of current layer), to store weights.
            # Size of the next layer, to store biases.
            total_size += sizes[i + 1] * sizes[i] + sizes[i + 1]

        # Weights and biases have a random normal distribution.
        return RNG.standard_normal(total_size) * GENOME.WEIGHTS_STD

    def get_layer_weights(self) -> list[tuple[NDArray[float], NDArray[float]]]:

        """
        Returns the weights and biases of each layer of the genome.

        Returns
        -------
        list[tuple[NDArray[float], NDArray[float]]]
            A list of tuples containing the weights and biases for each layer.
        """

        # Initialises the results list and offset.
        result: list[tuple[NDArray[float], NDArray[float]]] = []
        offset: int = 0

        # Computes a list of sizes (input + hidden layers + output).
        sizes: list[int] = [self.input_size] + self.topology + [self.output_size]

        # For each layer...
        for i in range(len(sizes) - 1):

            # The shape of the weights is (size of next layer, size of current layer).
            # Each row represents one neuron in the next layer.
            # Each neuron of the next layer needs the weights from ALL neurons of this layer.
            w_shape: tuple[int, int] = (sizes[i + 1], sizes[i])

            # Calculates the amount of weights and biases.
            w_size: int = sizes[i + 1] * sizes[i]
            b_size: int = sizes[i + 1]

            # Slices the data into a weight array and adds its size to the offset.
            W: NDArray[float] = self.weights[offset:offset + w_size].reshape(w_shape)
            offset += w_size

            # Slices the data into a bias array and adds its size to the offset.
            b: NDArray[float] = self.weights[offset:offset + b_size]
            offset += b_size

            # Appends a tuple of weights and biases of the current layer.
            result.append((W, b))

        return result

    def mutate(self) -> None:

        """
        Mutates the genome.
        Mutations may be applied to weights, activation functions, and/or topology.
        """

        # Mutates the weights.
        self._mutate_weights()

        # Mutates the activations.
        self._mutate_activations()

        # Mutates the topology.
        self._mutate_topology()

    def _mutate_weights(self) -> None:

        """
        Mutates the genome's weights.

        Notes
        -----
        A boolean mask is created to decide which weights to mutate based on ``MUTATION_CHANCE_WEIGHT``.

        A noise factor within ``[-MUTATION_NOISE_LIMIT, MUTATION_NOISE_LIMIT]`` is applied to the selected weights.
        """

        # Creates a mask that dictates which weights to change.
        weights_mask: NDArray[bool] = RNG.uniform(size=self.weights.shape) < MUTATION.CHANCE_WEIGHT

        # Creates an array with random noises (some might go unused depending on the mask).
        noise: NDArray[float] = RNG.uniform(-MUTATION.NOISE_LIMIT, MUTATION.NOISE_LIMIT, size=self.weights.shape)

        # Adds the noise to the selected weights.
        self.weights[weights_mask] += noise[weights_mask]

    def _mutate_activations(self) -> None:

        """
        Mutates the genome's activation functions.

        Notes
        -----
        This loops through all activation functions except the output layer's (which should stay as sigmoid).

        It chooses random activations to mutate based on ``MUTATION_CHANCE_ACTIVATION``.
        """

        # For each activation function (excluding the output's)...
        for i in range(len(self.activations) - 1):

            # If it should be mutated, select a new, random activation function.
            if RNG.uniform() < MUTATION.CHANCE_ACTIVATION:
                self.activations[i] = Genome._random_activation()

    def _mutate_topology(self) -> None:

        """
        Mutates the genome's topology.

        Notes
        -----
        A mutation only occurs based on ``MUTATION_CHANCE_TOPOLOGY``.

        A random mutation is chosen between adding a layer, removing a layer, and resizing a layer.
        """

        # Checks whether to mutate the topology.
        if RNG.uniform() >= MUTATION.CHANCE_TOPOLOGY:
            return

        # Selects a random topology mutation.
        mutation: TopologyMutation = RNG.choice(list(TopologyMutation))

        match mutation:

            case TopologyMutation.ADD:
                self._add_layer()
            case TopologyMutation.REMOVE:
                self._remove_layer()
            case TopologyMutation.RESIZE:
                self._resize_layer()

        return

    def _add_layer(self) -> None:

        """
        Adds a new layer to the genome at a random position.
        The new layer has a random size and activation function.
        All weights are recomputed afterward.

        Notes
        -----
        The size of the hidden layer is bound to ``[GENOME_MIN_NEURONS, GENOME_MAX_NEURONS]``.

        If the number of layers is already equal to ``GENOME_MAX_LAYERS``, a new layer is
        not added.
        """

        # Checks whether a new layer can be added.
        if len(self.topology) >= GENOME.MAX_LAYERS:
            return

        # Selects a random index (including the non-existent final index).
        index: int = int(RNG.integers(0, len(self.topology), endpoint=True))

        # Selects the size for the new layer.
        new_size: int = int(RNG.integers(GENOME.MIN_NEURONS, GENOME.MAX_NEURONS, endpoint=True))
        self.topology.insert(index, new_size)

        # Selects the activation function for the new layer.
        self.activations.insert(index, Genome._random_activation())

        # Recomputes weights.
        self.weights = Genome._random_weights(self.input_size, self.topology, self.output_size)

    def _remove_layer(self) -> None:

        """
        Removes a randomly selected layer from the genome.
        All weights are recomputed afterward.

        Notes
        -----
        If the number of layers is already equal to ``GENOME_MIN_LAYERS``, no layer
        is removed.
        """

        # Checks whether a layer can be removed.
        if len(self.topology) <= GENOME.MIN_LAYERS:
            return

        # Selects a random index.
        index: int = int(RNG.integers(0, len(self.topology)))

        # Removes the topology and activation function at that index.
        self.topology.pop(index)
        self.activations.pop(index)

        # Recomputes weights.
        self.weights = Genome._random_weights(self.input_size, self.topology, self.output_size)

    def _resize_layer(self) -> None:

        """
        Resizes a randomly selected hidden layer.
        All weights are recomputed afterward.

        Notes
        -----
        The size of the hidden layer is always bound to ``[GENOME_MIN_NEURONS, GENOME_MAX_NEURONS]``.

        The change in size is within ``[-MUTATION_RESIZE_LIMIT, MUTATION_RESIZE_LIMIT]`` .
        """

        # Selects a random topology's index.
        index: int = int(RNG.integers(0, len(self.topology)))

        # Creates a random size delta (difference) and applies it to the topology, keeping it within bounds.
        size_delta: int = int(RNG.integers(-MUTATION.RESIZE_LIMIT, MUTATION.RESIZE_LIMIT, endpoint=True))
        self.topology[index] = np.clip(self.topology[index] + size_delta, GENOME.MIN_NEURONS, GENOME.MAX_NEURONS)

        # Recomputes weights.
        self.weights = Genome._random_weights(self.input_size, self.topology, self.output_size)

    def copy(self) -> Genome:

        """
        Creates a copy of the genome.
        Input and output sizes, topology, activation function, and weights
        are copied.

        Returns
        -------
        Genome
            A copy of the genome.
        """

        # Creates a copy of the topology, activation functions, and weights.
        topology: list[int] = self.topology.copy()
        activations: list[ActivationFunction] = self.activations.copy()  # Shallow copy is fine.
        weights: NDArray[float] = self.weights.copy()

        return Genome(self.input_size, self.output_size, topology, activations, weights)


class TopologyMutation(Enum):

    ADD = 0
    REMOVE = 1
    RESIZE = 2
