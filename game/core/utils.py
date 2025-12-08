import pygame

from pygame.surface import Surface
from pygame.rect import Rect
from typing import Tuple


def draw_outlined_text(
        screen: Surface,
        text: str,
        pos: Tuple[int, int],
        text_color: Tuple[int, int, int] = (255, 255, 255),
        outline_color: Tuple[int, int, int] = (0, 0, 0),
        outline_thickness: int = 2,
        font_size: int = 24,
        align: str = "center"
) -> None:
    """
    Draws text with an outline at a given position.

    Parameters
    ----------
    screen : Surface
        The surface to draw on.
    text : str
        The text to draw.
    pos : tuple[int, int]
        The center position to draw the text at.
    text_color : tuple[int, int, int], optional
        The main text color (defaults to white).
    outline_color : tuple[int, int, int], optional
        The outline color (defaults to black).
    outline_thickness : int, optional
        Thickness of the outline in pixels (defaults to 2).
    font_size : int, optional
        Font size to use.
    align : str, optional
        The alignment of the text.
    """

    # Default font.
    font = pygame.font.Font(None, font_size)

    # Renders the surfaces.
    outline_surf = font.render(text, True, outline_color)
    text_surf = font.render(text, True, text_color)

    # Defines a rect function based on alignment.
    def get_rect(surf: Surface, position: tuple[int, int]) -> Rect:
        if align == "left":
            return surf.get_rect(topleft=position)
        return surf.get_rect(center=position)

    # Draws the outline in 8 directions.
    for dx in [-outline_thickness, 0, outline_thickness]:
        for dy in [-outline_thickness, 0, outline_thickness]:
            if dx != 0 or dy != 0:
                screen.blit(outline_surf, get_rect(outline_surf, (pos[0] + dx, pos[1] + dy)))

    # Draws the main text.
    screen.blit(text_surf, get_rect(text_surf, pos))
