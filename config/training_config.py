from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:

    """
    Contains training-related configuration variables.
    """

    POPULATION_SIZE: int = 50
    MAX_GENERATION_TIME: float = 30.0
    SPEED: int = 10
    INTERVAL: int = 4
    SAVE_AMOUNT: int = 10
    AUTOSAVE_INTERVAL: int = 25


@dataclass(frozen=True)
class FitnessConfig:

    """
    Contains fitness-related configuration variables.
    """

    REWARD_DISTANCE: float = 2.0
    REWARD_SAFETY: float = 50.0
    REWARD_VELOCITY: float = 0.1
    REWARD_CHECKPOINT: float = 1000.0
    REWARD_LAP: float = 10000.0

    PENALTY_WRONG_CHECKPOINT: float = -10000.0
    PENALTY_TIME: float = -200.0


TRAINING = TrainingConfig()
FITNESS = FitnessConfig()
