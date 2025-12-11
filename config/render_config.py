from dataclasses import dataclass
from typing import ClassVar
from pygame import Color


@dataclass(frozen=True)
class ColoursConfig:

    """
    Contains colour-related configuration variables.
    """

    BACKGROUND: ClassVar[Color] = Color(0, 0, 0)

    ITEM_SELECTED: ClassVar[Color] = Color(40, 80, 40)
    ITEM_UNSELECTED: ClassVar[Color] = Color(30, 30, 30)

    BUTTON: ClassVar[Color] = Color(0, 0, 0)
    BUTTON_HOVERED: ClassVar[Color] = Color(30, 30, 30)
    BUTTON_BORDER: ClassVar[Color] = Color(255, 255, 255)
    BUTTON_ENABLED: ClassVar[Color] = Color(0, 0, 0)
    BUTTON_DISABLED_ALPHA: int = 100

    TEXT_MAIN: ClassVar[Color] = Color(255, 255, 255)
    TEXT_SECONDARY: ClassVar[Color] = Color(100, 100, 100)
    TEXT_ERROR: ClassVar[Color] = Color(150, 100, 100)
    TEXT_SELECTED: ClassVar[Color] = Color(150, 255, 150)


@dataclass(frozen=True)
class FontsConfig:

    """
    Contains font-related configuration variables.
    """

    PATH: str = 'data/fonts/petty_5x5.otf'

    SIZE_NORMAL: int = 10
    SIZE_LARGE: int = 15
    SIZE_XL: int = 20


COLOURS = ColoursConfig()
FONTS = FontsConfig()
