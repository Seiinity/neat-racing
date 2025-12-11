from dataclasses import dataclass
from pygame import Color


@dataclass(frozen=True)
class ColoursConfig:

    """
    Contains colour-related configuration variables.
    """

    BACKGROUND = Color(0, 0, 0)
    BUTTON_DISABLED = Color(20, 20, 20)


COLOURS = ColoursConfig()
