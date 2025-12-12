import pygame

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class GameConfig:

    """
    Contains game-related configuration variables.
    """

    SCREEN_WIDTH: int = 1280
    SCREEN_HEIGHT: int = 720

    FPS: int = 60
    FIXED_DT: float = 1 / FPS


@dataclass(frozen=True)
class InputConfig:

    """
    Contains input-related configuration variables.
    """

    KEY_ACCELERATE: int = pygame.K_w
    KEY_BRAKE: int = pygame.K_s
    KEY_TURN_LEFT: int = pygame.K_a
    KEY_TURN_RIGHT: int = pygame.K_d

    KEY_CHECKPOINTS: int = pygame.K_0
    KEY_SENSORS: int = pygame.K_1


@dataclass(frozen=True)
class CarConfig:

    """
    Contains car-related configuration variables.
    """

    SIZE: int = 20
    ACCELERATION: float = 500
    BRAKE_STRENGTH: float = 600
    TURN_SPEED: float = 400
    FRICTION: float = 0.98
    SLIDING_FRICTION: float = 0.8

    SHAPE: ClassVar[dict] = {
        'triangle': [(0.75, 0), (-0.5, -0.5), (-0.5, 0.5)],
        'line': [(-2 / 3, -0.5), (-2 / 3, 0.5)]
    }

    SENSORS: ClassVar[list[float]] = [-45, -30, -15, 0, 15, 30, 45]
    SENSOR_RANGE = 100


@dataclass(frozen=True)
class TrackConfig:

    CHECKPOINT_COLOUR: tuple[int, int, int] = (255, 255, 0)


GAME: GameConfig = GameConfig()
INPUT: InputConfig = InputConfig()
CAR: CarConfig = CarConfig()
TRACK: TrackConfig = TrackConfig()
