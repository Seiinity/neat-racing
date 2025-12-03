from dataclasses import dataclass

import pygame

@dataclass(frozen=True)
class CarConfig:

    """
    Contains car-related configuration variables.
    """

    SIZE: int = 30
    ACCELERATION: float = 700
    BRAKE_STRENGTH: float = 800
    TURN_SPEED: float = 4
    FRICTION: float = 0.98

    SHAPE = {
        'triangle': [(0.75, 0), (-0.5, -0.5), (-0.5, 0.5)],
        'line': [(-2 / 3, -0.5), (-2 / 3, 0.5)]
    }

@dataclass(frozen=True)
class InputConfig:

    """
    Contains input-related configuration variables.
    """

    KEY_ACCELERATE: int = pygame.K_w
    KEY_BRAKE: int = pygame.K_s
    KEY_TURN_LEFT: int = pygame.K_a
    KEY_TURN_RIGHT: int = pygame.K_d

@dataclass(frozen=True)
class GameConfig:

    """
    Contains game-related configuration variables.
    """

    FPS = 60
    FIXED_DT = 1 / FPS

CAR = CarConfig()
INPUT = InputConfig()
GAME = GameConfig()