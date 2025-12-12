import pygame

from typing import Any, Optional
from pygame import Color, Rect, Surface
from config import COLOURS, FONTS
from src.core.utils import draw_outlined_text


class ListItem:

    """
    List item class for selectable list UI elements.
    Supports both single and multi-selection modes.
    """

    def __init__(self, x: int, y: int, width: int, height: int, text: str, data: Any = None) -> None:

        self.rect: Rect = Rect(x, y, width, height)
        self.text: str = text
        self.data: Any = data
        self.is_selected: bool = False
        self.is_hovered: bool = False
        self.selection_index: Optional[int] = None

    def draw(self, screen: Surface) -> None:

        """
        Draws the list item on the screen.

        Parameters
        ----------
        screen
            The screen to draw the list item on.
        """

        # Determines colors based on selection state.
        bg_colour: Color
        text_colour: Color

        if self.is_selected:
            bg_colour = COLOURS.ITEM_SELECTED
            text_colour = COLOURS.TEXT_SELECTED
        else:
            bg_colour = COLOURS.ITEM_UNSELECTED
            text_colour = COLOURS.TEXT_SECONDARY

        # Lightens the background color when hovered.
        if self.is_hovered and (self.selection_index or not self.is_selected):
            bg_colour = Color(
                min(255, bg_colour.r + 20),
                min(255, bg_colour.g + 20),
                min(255, bg_colour.b + 20),
                bg_colour.a
            )

        # Draws the background.
        pygame.draw.rect(screen, bg_colour, self.rect, border_radius=5)

        # Determines indicator text based on selection mode.
        if self.selection_index is not None:
            indicator_text: str = f"[{self.selection_index}]"
        else:
            indicator_text = "[X]" if self.is_selected else "[ ]"

        text_position: int = self.rect.x + 15

        # Draws the selection indicator.
        draw_outlined_text(
            screen,
            indicator_text,
            (text_position, self.rect.y + 9),
            align='left',
            font_size=FONTS.SIZE_NORMAL,
            text_colour=text_colour
        )

        # Draws the text.
        draw_outlined_text(
            screen,
            self.text,
            (text_position + 50, self.rect.y + 9),
            align='left',
            font_size=FONTS.SIZE_NORMAL,
            text_colour=text_colour
        )

    def handle_event(self, event: pygame.event.Event) -> bool:

        """
        Handles mouse events for the list item.

        Parameters
        ----------
        event : pygame.event.Event
            The pygame event to handle.

        Returns
        -------
        bool
            ``True`` if the item was clicked, ``False`` otherwise.
        """

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
