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

    CHANCE_WEIGHT: float = 0.01
    CHANCE_TOPOLOGY: float = 0.01
    CHANCE_ACTIVATION: float = 0.02
    NOISE_LIMIT: float = 0.05
    RESIZE_LIMIT: int = 4


@dataclass(frozen=True)
class GeneticConfig:

    """
    Contains the configuration variables for the genetic algorithm.
    """

    ELITISM_CUTOFF: int = 5

    REWARD_CHECKPOINT: int = 1000
    REWARD_LAP: int = 10000
    REWARD_TIME: int = 10
    REWARD_DEATH: int = -500

    POPULATION_SIZE: int = 50
    MAX_GENERATION_TIME: float = 60.0


GENOME = GenomeConfig()
MUTATION = MutationConfig()
GENETIC = GeneticConfig()
