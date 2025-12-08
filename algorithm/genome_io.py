import pickle

from typing import Type, TypedDict
from pathlib import Path
from numpy.typing import NDArray
from algorithm.genome import Genome
from algorithm.activation_function import ActivationFunction, ReLU, Sigmoid, Tanh, Softmax


class GenomeIO:

    """
    Utilities for saving and loading genomes.
    """

    @staticmethod
    def save_genome(genome: Genome, filepath: str) -> None:

        """
        Saves a genome to a file.

        Parameters
        ----------
        genome : Genome
            The genome to save.
        filepath : str
            Path to save the genome to.
        """

        # Creates the directory if it doesn't exist.
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Prepares the data for serialisation.
        data: GenomeData = {
            'input_size': genome.input_size,
            'output_size': genome.output_size,
            'topology': genome.topology,
            'activations': [type(act).__name__ for act in genome.activations],
            'weights': genome.weights
        }

        # Writes the file.
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)  # type: ignore (false alarm!)

        print(f"Genome saved to: {filepath}")

    @staticmethod
    def load_genome(filepath: str) -> Genome:

        """
        Loads a genome from a file.

        Parameters
        ----------
        filepath : str
            Path to load the genome from.

        Returns
        -------
        Genome
            The loaded genome.
        """

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Maps activation function names to their type.
        activation_map: dict[str, Type[ActivationFunction]] = {
            'ReLU': ReLU,
            'Sigmoid': Sigmoid,
            'Tanh': Tanh,
            'Softmax': Softmax
        }

        # Reconstructs activation functions.
        activations: list[ActivationFunction] = [
            activation_map[name]()
            for name in data['activations']
        ]

        # Reconstructs the genome.
        genome = Genome(
            input_size=data['input_size'],
            output_size=data['output_size'],
            topology=data['topology'],
            activations=activations,
            weights=data['weights']
        )

        print(f"Genome loaded from: {filepath}")
        return genome

    @staticmethod
    def save_best_genomes(genetic_algorithm, num_best: int, directory: str) -> None:

        """
        Saves the best genomes from a genetic algorithm.

        Parameters
        ----------
        genetic_algorithm : GeneticAlgorithm
            The genetic algorithm containing the population.
        num_best : int
            Number of genomes to save.
        directory : str
            Directory to save genomes to.
        """

        best_genomes = genetic_algorithm.get_top(num_best)

        for i, genome in enumerate(best_genomes, 1):
            filepath = f"{directory}/genome_gen{genetic_algorithm.generation}_rank{i}.pkl"
            GenomeIO.save_genome(genome, filepath)

        print(f"Saved top {num_best} genomes to {directory}")


class GenomeData(TypedDict):

    input_size: int
    output_size: int
    topology: list[int]
    activations: list[str]
    weights: NDArray[float]
