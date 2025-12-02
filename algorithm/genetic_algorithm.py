from algorithm.genome import Genome


class GeneticAlgorithm:

    def __init__(self, population_size: int, input_size: int, output_size: int):

        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size

        # Creates the initial population from random Genomes.
        self.population = [Genome.random(input_size, output_size) for _ in range(population_size)]