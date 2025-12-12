from collections.abc import Callable
from config import GENETIC, RNG
from .genome import Genome


class GeneticAlgorithm:

    """
    Represents a genetic algorithm.

    Attributes
    ----------
    generation : int
        The current generation.
    population : list[tuple[Genome, float]]
        A list of tuples containing the genomes and their fitness score.

    Methods
    -------
    get_top(num: int) -> list[Genome]
        Returns the top genomes in the population.
    next_generation(fitness_func: Callable[[Genome], float]) -> None
        Creates the next generation of genomes.
    """

    def __init__(
        self,
        population_size: int,
        input_size: int,
        output_size: int
    ) -> None:

        self.generation: int = 0
        self.population: list[tuple[Genome, float]] = [
            (Genome.random(input_size, output_size), 0) for _ in range(population_size)
        ]

        self._population_size: int = population_size

    def get_top(self, num: int) -> list[Genome]:

        """
        Retrieves the top genomes from the population.

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

        Parameters
        ----------
        fitness_func : Callable[[Genome], float]
            The fitness function used to evaluate the population.

        Notes
        -----
        The top ``ELITISM_CUTOFF`` genomes are directly copied to the new generation.
        The remaining genomes have a chance to be mutated.
        """

        self._evaluate_fitness(fitness_func)

        survivors: list[Genome] = self._select_survivors()

        # The elitism-chosen survivors get copied directly.
        new_population: list[tuple[Genome, float]] = [
            (genome.copy(), 0) for genome in survivors[:GENETIC.ELITISM_CUTOFF]
        ]

        # Tournament pool for parent selection.
        tournament_pool_size: int = max(
            int(len(self.population) * GENETIC.TOURNAMENT_SIZE),
            GENETIC.ELITISM_CUTOFF + 1
        )
        tournament_pool: list[tuple[Genome, float]] = self.population[:tournament_pool_size]

        # Fills the rest via crossover + mutation.
        while len(new_population) < self._population_size:

            parent_a, fitness_a = self._run_tournament_with_fitness(tournament_pool)
            parent_b, fitness_b = self._run_tournament_with_fitness(tournament_pool)

            # Ensures parent_a is the fitter one.
            if fitness_b > fitness_a:
                parent_a, parent_b = parent_b, parent_a

            child = self._crossover(parent_a, parent_b)
            child.mutate()
            new_population.append((child, 0))

        self.generation += 1
        self.population = new_population

    def _evaluate_fitness(self, fitness_func: Callable[[Genome], float]) -> None:

        """
        Evaluates the fitness of the population based on a fitness function.

        Parameters
        ----------
        fitness_func : Callable[[Genome], float]
            The fitness function used to evaluate the population.

        Notes
        -----
        After evaluation, a fitness score is attributed to each genome.
        """

        self.population = [(genome, fitness_func(genome)) for genome, _ in self.population]

    def _select_survivors(self) -> list[Genome]:

        """
        Picks the best genomes out of the population.

        Returns
        -------
        list[Genome]
            A list of all surviving genomes.

        Notes
        -----
        The top ``GeneticConfig.ELITISM_CUTOFF`` genomes are
        picked directly. The top ``GeneticConfig.TOURNAMENT_SIZE`` (%)
        of genomes go through tournament selection.
        """

        # Sorts the population by fitness (descending) and keeps the best ones.
        self.population.sort(key=lambda x: x[1], reverse=True)
        survivors: list[Genome] = [genome for genome, _ in self.population[:GENETIC.ELITISM_CUTOFF]]

        # Allows tournament selection from the top n% of performers.
        tournament_pool_size: int = max(
            int(len(self.population) * GENETIC.TOURNAMENT_SIZE),
            GENETIC.ELITISM_CUTOFF + 1
        )
        tournament_pool: list[tuple[Genome, float]] = self.population[:tournament_pool_size]

        # Runs the tournaments.
        for _ in range(self._population_size - GENETIC.ELITISM_CUTOFF):

            winner: Genome = GeneticAlgorithm._run_tournament(tournament_pool)
            survivors.append(winner)

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

        # If there's only one genome, returns it.
        if len(genomes) < 2:
            return genomes[0][0]

        # Picks two random genomes keeps the best one.
        picked: list[tuple[Genome, float]] = RNG.choice(genomes, size=2).tolist()
        winner: tuple[Genome, float] = max(picked, key=lambda x: x[1])

        return winner[0]

    @staticmethod
    def _run_tournament_with_fitness(genomes: list[tuple[Genome, float]]) -> tuple[Genome, float]:

        """
        Runs a tournament and returns both the winner and its fitness.
        """

        if len(genomes) < 2:
            return genomes[0]

        picked: list[tuple[Genome, float]] = RNG.choice(genomes, size=2).tolist()
        winner: tuple[Genome, float] = max(picked, key=lambda x: x[1])

        return winner

    @staticmethod
    def _crossover(parent_a: Genome, parent_b: Genome) -> Genome:

        """
        Creates a child genome via arithmetic crossover.

        Parameters
        ----------
        parent_a : Genome
            The first parent genome.
        parent_b
            The second parent genome.

        Returns
        -------
        Genome
            A child genome.
        """

        # Only crossover if topologies match.
        if parent_a.topology != parent_b.topology:
            return parent_a.copy()

        # Performs the arithmetic crossover of weights.
        alpha = RNG.uniform(0.0, 1.0, size=parent_a.weights.shape)
        child_weights = alpha * parent_a.weights + (1 - alpha) * parent_b.weights

        return Genome(
            parent_a.input_size,
            parent_a.output_size,
            parent_a.topology.copy(),
            parent_a.activations.copy(),
            child_weights
        )
