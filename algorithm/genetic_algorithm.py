from collections.abc import Callable
from algorithm.genome import Genome
from algorithm.config import ELITISM_CUTOFF
from rng import rng

class GeneticAlgorithm:

    """
    Represents a genetic algorithm.

    Attributes
    ----------
    population_size : int
        The number of genomes in the population.
    input_size : int
        The number of neurons in the input layer.
    output_size : int
        The number of neurons in the output layer.
    generation : int
        The current generation.
    population : list[tuple[Genome, float]]
        A list of tuples containing the genomes and their fitness score.

    Methods
    -------
    get_top(self, num: int) -> list[Genome]
        Returns the top genomes in the population.
    next_generation(self, fitness_func: Callable[[Genome], float]) -> None
        Creates the next generation of genomes.
    """

    def __init__(self, population_size: int, input_size: int, output_size: int) -> None:

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.population_size: int = population_size
        self.generation: int = 0

        # Creates the initial population from random Genomes and a fitness of 0.
        self.population: list[tuple[Genome, float]] = [(Genome.random(input_size, output_size), 0) for _ in range(population_size)]

    def get_top(self, num: int) -> list[Genome]:

        """
        Returns the top genomes from the population.

        Parameters
        ----------
        num : int
            The number of genomes to return.

        Returns
        -------
        list[Genome]
            A list with the top genomes from the population.
        """

        self.population.sort(key=lambda x: x[1], reverse=True)
        return [genome for genome, _ in self.population[:num]]

    def next_generation(self, fitness_func: Callable[[Genome], float]) -> None:

        """
        Creates the next generation of genomes.

        Notes
        -----
        The top ``ELITISM_CUTOFF`` genomes are directly copied to the new generation.
        The remaining genomes have a chance to be mutated.

        Parameters
        ----------
        fitness_func : Callable[[Genome], float]
            The fitness function used to evaluate the population.

        """

        self._evaluate_fitness(fitness_func)
        survivors: list[Genome] = self._select_survivors()

        # The elitism-chosen survivors get copied directly, the remaining ones have a chance to mutate.
        new_population: list[tuple[Genome, float]] = [(genome.copy(), 0) for genome in survivors[:ELITISM_CUTOFF]]

        # The remaining ones have a chance to mutate.
        for genome in survivors[ELITISM_CUTOFF:]:
            copy = genome.copy()
            copy.mutate()
            new_population.append((copy, 0))

        self.generation += 1
        self.population = new_population

    def _evaluate_fitness(self, fitness_func: Callable[[Genome], float]) -> None:

        """
        Evaluates the fitness of the population based on a fitness function.

        Notes
        -----
        After evaluation, a fitness score is attributed to each genome.

        Parameters
        ----------
        fitness_func : Callable[[Genome], float]
            The fitness function used to evaluate the population.

        """

        self.population = [(genome, fitness_func(genome)) for genome, _ in self.population]

    def _select_survivors(self) -> list[Genome]:

        """
        Picks the best genomes out of the population.

        Notes
        -----
        The top ``ELITISM_CUTOFF`` genomes are picked directly.
        The remaining genomes go through tournament selection. This selection allows
        for duplicate genomes, which is desirable for natural selection.

        Returns
        -------
        list[Genome]
            A list of all surviving genomes.
        """

        # Sorts the population by fitness (descending) and keeps the best ones.
        self.population.sort(key=lambda x: x[1], reverse=True)
        survivors: list[Genome] = [genome for genome, _ in self.population[:ELITISM_CUTOFF]]
        remaining_genomes: list[tuple[Genome, float]] = self.population[ELITISM_CUTOFF:]

        # Runs tournaments on the remaining genomes.
        for _ in range(self.population_size - ELITISM_CUTOFF):
            survivors.append(GeneticAlgorithm._run_tournament(remaining_genomes))

        return survivors

    @staticmethod
    def _run_tournament(genomes: list[tuple[Genome, float]]) -> Genome:

        """
        Runs a tournament between two genomes to find the fittest one.
        If only one genome is passed, it is returned back.

        Parameters
        ----------
        genomes : list[tuple[Genome, float]]
            A list of (genome, fitness) tuples.

        Returns
        -------
        Genome
            The genome with the highest fitness.
        """

        # If there's only one genome, return it.
        if len(genomes) < 2:
            return genomes[0][0]

        # Picks two random genomes keeps the best one.
        picked: list[tuple[Genome, float]] = rng.choice(genomes, size=2).tolist()
        winner: tuple[Genome, float] = max(picked, key=lambda x: x[1])

        return winner[0]