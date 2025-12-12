from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ControllerConfig:

    """
    Contains AI controller-related configuration variables.
    """

    SENSORS: ClassVar[list[float]] = [-45, -30, -15, 0, 15, 30, 45]
    SENSOR_RANGE = 100


@dataclass(frozen=True)
class TrainingConfig:

    """
    Contains training-related configuration variables.
    """

    POPULATION_SIZE: int = 50
    MAX_GENERATION_TIME: float = 30.0
    SPEED: int = 20
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


CONTROLLER: ControllerConfig = ControllerConfig()
TRAINING: TrainingConfig = TrainingConfig()
FITNESS: FitnessConfig = FitnessConfig()
