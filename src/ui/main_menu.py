import pygame

from pygame import Surface
from pygame.time import Clock
from config import COLOURS, FONTS, GAME
from src.core.utils import draw_outlined_text
from .button import Button


class MainMenu:

    """
    Main menu screen with clickable options.

    Attributes
    ----------
    screen : Surface
        The Pygame display surface.
    clock : Clock
        The Pygame clock for timing.
    running : bool
        Whether the menu is currently running.
    selected_mode : str | None
        The mode selected by the user, or None if not yet selected.
    """

    def __init__(self) -> None:

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((GAME.SCREEN_WIDTH, GAME.SCREEN_HEIGHT))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.selected_mode: str | None = None

        pygame.display.set_caption("NEAT-ish Racing")

        # Resets the cursor.
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        # Creates the menu buttons.
        button_width: int = 150
        button_height: int = 40
        button_x: int = (GAME.SCREEN_WIDTH - button_width) // 2

        self._train_button: Button = Button(
            x=button_x,
            y=330,
            width=button_width,
            height=button_height,
            text="Train AI"
        )

        self._play_button: Button = Button(
            x=button_x,
            y=380,
            width=button_width,
            height=button_height,
            text="Play"
        )

        self._quit_button: Button = Button(
            x=button_x,
            y=430,
            width=button_width,
            height=button_height,
            text="Quit"
        )

    def run(self) -> str | None:

        """
        Runs the main menu loop.

        Returns
        -------
        str | None
            The selected mode ('train', 'play'), or None if the menu was closed.
        """

        while self.running:

            self._process_events()
            self._draw()

            self.clock.tick(60)

            if self.selected_mode is not None:
                break

        return self.selected_mode

    def _process_events(self) -> None:

        """
        Processes all pending Pygame events.
        """

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

            # Handles button clicks.
            if self._train_button.handle_event(event):
                self.selected_mode = 'train'

            if self._play_button.handle_event(event):
                self.selected_mode = 'play'

            if self._quit_button.handle_event(event):
                self.running = False

    def _draw(self) -> None:

        """
        Draws the menu screen.
        """

        # Background.
        self.screen.fill(COLOURS.BACKGROUND)

        # Title.
        draw_outlined_text(
            self.screen,
            "NEAT-ish Racing",
            (GAME.SCREEN_WIDTH // 2, 150),
            font_size=FONTS.SIZE_XL
        )

        # Subtitle.
        draw_outlined_text(
            self.screen,
            "A Neuroevolution Racing Game",
            (GAME.SCREEN_WIDTH // 2, 190),
            font_size=FONTS.SIZE_LARGE,
            text_colour=COLOURS.TEXT_SECONDARY
        )

        # Buttons.
        self._train_button.draw(self.screen)
        self._play_button.draw(self.screen)
        self._quit_button.draw(self.screen)

        # Footer.
        draw_outlined_text(
            self.screen,
            "Press ESC to quit",
            (GAME.SCREEN_WIDTH // 2, 600),
            font_size=FONTS.SIZE_NORMAL,
            text_colour=COLOURS.TEXT_SECONDARY
        )

        pygame.display.flip()
