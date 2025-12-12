from dataclasses import dataclass


@dataclass(frozen=True)
class GenomeConfig:

    """
    Contains genome-related configuration variables.
    """

    MIN_LAYERS: int = 1
    MAX_LAYERS: int = 4
    MIN_NEURONS: int = 5
    MAX_NEURONS: int = 10
    WEIGHTS_STD: float = 0.1


@dataclass(frozen=True)
class MutationConfig:

    """
    Contains mutation-related configuration variables.
    """

    CHANCE_WEIGHT: float = 0.1
    CHANCE_TOPOLOGY: float = 0.05
    CHANCE_ACTIVATION: float = 0.05
    NOISE_LIMIT: float = 0.1
    RESIZE_LIMIT: int = 4


@dataclass(frozen=True)
class GeneticConfig:

    """
    Contains the configuration variables for the genetic algorithm.
    """

    ELITISM_CUTOFF: int = 5
    TOURNAMENT_SIZE: float = 1


GENOME: GenomeConfig = GenomeConfig()
MUTATION: MutationConfig = MutationConfig()
GENETIC: GeneticConfig = GeneticConfig()
