from dataclasses import dataclass
import pygame

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


@dataclass(frozen=True)
class CarConfig:

    """
    Contains car-related configuration variables.
    """

    SIZE: int = 20
    ACCELERATION: float = 500
    BRAKE_STRENGTH: float = 600
    TURN_SPEED: float = 4
    FRICTION: float = 0.98

    SHAPE = {
        'triangle': [(0.75, 0), (-0.5, -0.5), (-0.5, 0.5)],
        'line': [(-2 / 3, -0.5), (-2 / 3, 0.5)]
    }


@dataclass(frozen=True)
class TrackConfig:

    CHECKPOINT_COLOUR: tuple[int, int, int] = (255, 255, 0)


GAME = GameConfig()
INPUT = InputConfig()
CAR = CarConfig()
TRACK = TrackConfig()