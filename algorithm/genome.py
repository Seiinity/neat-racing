from __future__ import annotations

import numpy as np
import algorithm.config as cfg

from numpy.typing import NDArray
from algorithm.activation_function import ActivationFunction, ReLU, Sigmoid, Tanh, Softmax

class Genome:

    def __init__(self, input_size: int, output_size: int, topology: list[int], activations: list[ActivationFunction], weights: NDArray[float]):

        self.input_size = input_size
        self.output_size = output_size
        self.topology: list[int] = topology
        self.activations: list[ActivationFunction] = activations
        self.weights: NDArray[float] = weights

    @staticmethod
    def random(input_size: int, output_size: int) -> Genome:

        num_hidden_layers: int = np.random.randint(cfg.GENOME_MIN_LAYERS, cfg.GENOME_MAX_LAYERS + 1)
        topology: list[int] = np.random.randint(cfg.GENOME_MIN_NEURONS, cfg.GENOME_MAX_NEURONS + 1, size=num_hidden_layers).tolist()

        activations: list[ActivationFunction] = Genome._random_activations(num_hidden_layers)
        weights: NDArray[float] = Genome._random_weights(input_size, topology, output_size)

        return Genome(input_size, output_size, topology, activations, weights)

    @staticmethod
    def _random_activation() -> ActivationFunction:

        possible_activations: list[type[ActivationFunction]] = [ReLU, Sigmoid, Tanh]
        return np.random.choice(possible_activations)()

    @staticmethod
    def _random_activations(num_layers: int) -> list[ActivationFunction]:

        activations = [Genome._random_activation() for _ in range(num_layers)]
        activations.append(Softmax())
        return activations

    @staticmethod
    def _random_weights(input_size: int, topology: list[int], output_size: int) -> NDArray[float]:

        total_size: int = 0
        sizes: list[int] = [input_size] + topology + [output_size]

        for i in range(len(sizes) - 1):
            total_size += sizes[i + 1] * sizes[i] + sizes[i + 1]

        return np.random.randn(total_size) * cfg.GENOME_WEIGHTS_STD

    def get_layer_weights(self) -> list[tuple[NDArray[float], NDArray[float]]]:

        result = []
        sizes = [self.input_size] + self.topology + [self.output_size]
        offset = 0

        for i in range(len(sizes) - 1):

            w_shape = (sizes[i + 1], sizes[i])
            w_size = sizes[i + 1] * sizes[i]
            b_size = sizes[i + 1]

            W = self.weights[offset:offset + w_size].reshape(w_shape)
            offset += w_size

            b = self.weights[offset:offset + b_size]
            offset += b_size

            result.append((W, b))

        return result

    def mutate(self):

        # Mutates the weights.
        self._mutate_weights()

        # Mutates the activations.
        self._mutate_activations()

        # Mutates the topology.
        self._mutate_topology()

    def _mutate_weights(self) -> None:

        """
        Mutates the genome's weights.
        This creates a boolean mask to decide which weights to mutate based on MUTATION_CHANCE_WEIGHT.
        A noise factor within [-MUTATION_NOISE_LIMIT, MUTATION_NOISE_LIMIT] is applied to the selected weights.
        """

        weights_mask = np.random.uniform(size=self.weights.shape) < cfg.MUTATION_CHANCE_WEIGHT
        noise = np.random.uniform(-cfg.MUTATION_NOISE_LIMIT, cfg.MUTATION_NOISE_LIMIT, size=self.weights.shape)
        self.weights[weights_mask] += noise[weights_mask]

    def _mutate_activations(self) -> None:

        """
        Mutates the genome's activation functions.
        This loops through all activation functions except output (which should stay as Softmax).
        It chooses random activations to mutate based on MUTATION_CHANCE_ACTIVATION.
        """

        for i in range(len(self.activations) - 1):

            if np.random.uniform() < cfg.MUTATION_CHANCE_ACTIVATION:
                self.activations[i] = Genome._random_activation()

    def _mutate_topology(self) -> None:

        return