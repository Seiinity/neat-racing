import pygame

from pygame import Color, Rect, Surface
from config import COLOURS
from src.core.utils import draw_outlined_text


class Button:
    """
    Button class for UI interactions.
    """

    def __init__(self, x: int, y: int, width: int, height: int, text: str, disabled: bool = False) -> None:

        self.rect: Rect = Rect(x, y, width, height)
        self.text: str = text
        self.colour: Color = COLOURS.BUTTON
        self.is_hovered: bool = False
        self.disabled: bool = disabled

    def draw(self, screen: Surface) -> None:

        self.colour = COLOURS.BUTTON_HOVERED if self.is_hovered and not self.disabled else COLOURS.BUTTON

        if self.disabled:

            # Creates a temporary surface with per-pixel alpha.
            temp_surface: Surface = Surface(self.rect.size, pygame.SRCALPHA)

            # Draws the button components on the temporary surface.
            pygame.draw.rect(temp_surface, self.colour, temp_surface.get_rect(), border_radius=5)
            pygame.draw.rect(temp_surface, COLOURS.BUTTON_BORDER, temp_surface.get_rect(), 2, border_radius=5)

            # Draw the text.
            draw_outlined_text(temp_surface, self.text, temp_surface.get_rect().center)

            # Sets opacity to 50% and draws to the screen.
            temp_surface.set_alpha(COLOURS.BUTTON_DISABLED_ALPHA)
            screen.blit(temp_surface, self.rect.topleft)

        else:

            # Draws a regular button.
            pygame.draw.rect(screen, self.colour, self.rect, border_radius=5)
            pygame.draw.rect(screen, COLOURS.BUTTON_BORDER, self.rect, 2, border_radius=5)
            draw_outlined_text(screen, self.text, self.rect.center)

    def handle_event(self, event: pygame.event.Event) -> bool:

        """
        Handles mouse events for the button.

        Parameters
        ----------
        event : pygame.event.Event
            The pygame event to handle.

        Returns
        -------
        bool
            ``True`` if the button was clicked, ``False`` otherwise.
        """

        if self.disabled:
            return False

        if event.type == pygame.MOUSEMOTION:

            was_hovered: bool = self.is_hovered
            self.is_hovered = self.rect.collidepoint(event.pos)

            # Updates the cursor based on hover state.
            if self.is_hovered and not was_hovered:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            elif not self.is_hovered and was_hovered:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                return True

        return False
