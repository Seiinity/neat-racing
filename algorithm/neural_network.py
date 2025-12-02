from __future__ import annotations

from numpy.typing import NDArray
from algorithm.activation_function import ActivationFunction
from algorithm.genome import Genome

class NeuralNetwork:

    def __init__(self, layers: list[DenseLayer]):

        self.layers: list[DenseLayer] = layers

    @staticmethod
    def from_genome(genome: Genome) -> NeuralNetwork:

        layer_weights = genome.get_layer_weights()
        sizes = genome.topology + [genome.output_size]

        layers = [
            DenseLayer(size, act, W=W, b=b)
            for size, act, (W, b) in zip(sizes, genome.activations, layer_weights)
        ]

        return NeuralNetwork(layers)

    def forward(self, X: NDArray[float]) -> NDArray[float]:

        x = X

        for layer in self.layers:
            x = layer.forward(x)

        return x

class DenseLayer:

    def __init__(self, size: int, activation: ActivationFunction, W: NDArray[float], b: NDArray[float]):

        self.size: int = size
        self.activation: ActivationFunction = activation
        self.W: NDArray[float] = W
        self.b: NDArray[float] = b

    def forward(self, x: NDArray[float]) -> NDArray[float]:

        z = x @ self.W.T + self.b
        return self.activation.forward(z)