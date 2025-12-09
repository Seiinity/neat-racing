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

    REWARD_DISTANCE: float = 2.0
    REWARD_SAFETY: float = 50.0
    REWARD_VELOCITY: float = 0.1
    REWARD_CHECKPOINT: float = 1000.0
    REWARD_LAP: float = 10000.0

    PENALTY_WRONG_CHECKPOINT: float = -10000.0
    PENALTY_TIME: float = -200.0

    POPULATION_SIZE: int = 50
    MAX_GENERATION_TIME: float = 30.0

    TRAINING_INTERVAL: int = 4


GENOME = GenomeConfig()
MUTATION = MutationConfig()
GENETIC = GeneticConfig()
