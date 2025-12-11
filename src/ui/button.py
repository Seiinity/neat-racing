import pygame

from pygame import Color, Rect, Surface
from src.core.utils import draw_outlined_text


class Button:

    """
    Button class for UI interactions.
    """

    def __init__(self, x: int, y: int, width: int, height: int, text: str, colour: Color):

        self.rect = Rect(x, y, width, height)
        self.text = text
        self.colour = colour
        self.is_hovered = False

    def draw(self, screen: Surface):

        pygame.draw.rect(screen, self.colour, self.rect, border_radius=5)
        pygame.draw.rect(screen, Color(200, 200, 200), self.rect, 2, border_radius=5)

        draw_outlined_text(screen, self.text, self.rect.center)

    def handle_event(self, event) -> bool:

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                return True

        return False
