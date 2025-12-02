from __future__ import annotations

import numpy as np
import algorithm.config as cfg

from numpy.random import Generator
from numpy.typing import NDArray
from algorithm.activation_function import ActivationFunction, ReLU, Sigmoid, Tanh, Softmax
from enum import Enum

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
    random(input_size, output_size) -> Genome
        Creates a genome with a random topology, activation functions, and weights.
    get_layer_weights() -> list[tuple[NDArray[float], NDArray[float]]]
        Returns the weights and biases of each layer of the genome.
    mutate()
        Applies mutations to weights, activations, and/or topology.
    """

    rng: Generator = np.random.default_rng(seed=cfg.RANDOM_SEED)

    def __init__(self, input_size: int, output_size: int, topology: list[int], activations: list[ActivationFunction], weights: NDArray[float]):

        self.input_size = input_size
        self.output_size = output_size
        self.topology: list[int] = topology
        self.activations: list[ActivationFunction] = activations
        self.weights: NDArray[float] = weights

    @staticmethod
    def random(input_size: int, output_size: int) -> Genome:

        """
        Creates a genome with a random topology, activation functions, and weights.

        Notes
        -----
        The number of hidden layers is derived from `GENOME_MIN_LAYERS`
        and `GENOME_MAX_LAYERS`.The number of neurons in each hidden layer is
        derived from `GENOME_MIN_NEURONS` and `GENOME_MAX_NEURONS`.

        The output layer's activation function is always softmax.

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

        """

        num_hidden_layers: int = int(Genome.rng.integers(cfg.GENOME_MIN_LAYERS, cfg.GENOME_MAX_LAYERS, endpoint=True))
        topology: list[int] = Genome.rng.integers(cfg.GENOME_MIN_NEURONS, cfg.GENOME_MAX_NEURONS, endpoint=True, size=num_hidden_layers).tolist()

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

        possible_activations: list[type[ActivationFunction]] = [ReLU, Sigmoid, Tanh]
        return Genome.rng.choice(possible_activations)()

    @staticmethod
    def _random_activations(num_layers: int) -> list[ActivationFunction]:

        """
        Randomly selects an activation function from ReLU, Sigmoid, and Tanh
        for each hidden layer of the genome.

        Notes
        -----
        A Softmax activation is always appended at the end for the output layer.

        Parameters
        ----------
        num_layers: int
            The number of hidden layers of the genome.

        Returns
        -------
        list[ActivationFunction]
            A list of randomly chosen activation function instances.
        """

        activations: list[ActivationFunction] = [Genome._random_activation() for _ in range(num_layers)]
        activations.append(Softmax())
        return activations

    @staticmethod
    def _random_weights(input_size: int, topology: list[int], output_size: int) -> NDArray[float]:

        """
        Creates a normal distribution of random weights for each layer of the genome.

        Notes
        -----
        The standard deviation of the normal distribution is controlled with `GENOME_WEIGHTS_STD`.

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
        """

        total_size: int = 0
        sizes: list[int] = [input_size] + topology + [output_size]

        for i in range(len(sizes) - 1):
            total_size += sizes[i + 1] * sizes[i] + sizes[i + 1]

        return Genome.rng.standard_normal(total_size) * cfg.GENOME_WEIGHTS_STD

    def get_layer_weights(self) -> list[tuple[NDArray[float], NDArray[float]]]:

        """
        Returns the weights and biases of each layer of the genome.

        Returns
        -------
        list[tuple[NDArray[float], NDArray[float]]]
            A list of tuples containing the weights and biases for each layer.
        """

        result: list[tuple[NDArray[float], NDArray[float]]] = []
        sizes: list[int] = [self.input_size] + self.topology + [self.output_size]
        offset: int = 0

        for i in range(len(sizes) - 1):

            w_shape: tuple[int, int] = (sizes[i + 1], sizes[i])
            w_size: int = sizes[i + 1] * sizes[i]
            b_size: int = sizes[i + 1]

            W: NDArray[float] = self.weights[offset:offset + w_size].reshape(w_shape)
            offset += w_size

            b: NDArray[float] = self.weights[offset:offset + b_size]
            offset += b_size

            result.append((W, b))

        return result

    def mutate(self):

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
        A boolean mask is created to decide which weights to mutate based on `MUTATION_CHANCE_WEIGHT`.
        A noise factor within ``[-MUTATION_NOISE_LIMIT, MUTATION_NOISE_LIMIT]`` is applied to the selected weights.
        """

        weights_mask: NDArray[bool] = Genome.rng.uniform(size=self.weights.shape) < cfg.MUTATION_CHANCE_WEIGHT
        noise: NDArray[float] = Genome.rng.uniform(-cfg.MUTATION_NOISE_LIMIT, cfg.MUTATION_NOISE_LIMIT, size=self.weights.shape)
        self.weights[weights_mask] += noise[weights_mask]

    def _mutate_activations(self) -> None:

        """
        Mutates the genome's activation functions.

        Notes
        -----
        This loops through all activation functions except the output layer's (which should stay as Softmax).
        It chooses random activations to mutate based on `MUTATION_CHANCE_ACTIVATION`.
        """

        for i in range(len(self.activations) - 1):

            if Genome.rng.uniform() < cfg.MUTATION_CHANCE_ACTIVATION:
                self.activations[i] = Genome._random_activation()

    def _mutate_topology(self) -> None:

        """
        Mutates the genome's topology.

        Notes
        -----
        A mutation only occurs based on `MUTATION_CHANCE_TOPOLOGY`.
        A random mutation is chosen between adding a layer, removing a layer, and resizing a layer.
        """

        if Genome.rng.uniform() >= cfg.MUTATION_CHANCE_TOPOLOGY:
            return

        mutation: TopologyMutation = Genome.rng.choice(list(TopologyMutation))

        match mutation:

            case TopologyMutation.ADD:
                self._add_layer()
            case TopologyMutation.REMOVE:
                self._remove_layer()
            case TopologyMutation.RESIZE:
                self._resize_layer()

        return

    def _add_layer(self):

        return

    def _remove_layer(self):

        return

    def _resize_layer(self):

        return

class TopologyMutation(Enum):

    ADD = 0
    REMOVE = 1
    RESIZE = 2