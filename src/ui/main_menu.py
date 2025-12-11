import pygame

from config import COLOURS
from pygame import Surface, Color
from pygame.time import Clock
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

        self.screen: Surface = pygame.display.set_mode((1280, 720))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.selected_mode: str | None = None

        pygame.display.set_caption("NEAT-ish Racing")

        # Creates the menu buttons.
        button_width: int = 300
        button_height: int = 70
        button_x: int = (1280 - button_width) // 2

        self._train_button: Button = Button(
            x=button_x,
            y=320,
            width=button_width,
            height=button_height,
            text="Train AI",
            colour=COLOURS.BACKGROUND
        )

        self._play_button: Button = Button(
            x=button_x,
            y=400,
            width=button_width,
            height=button_height,
            text="Play",
            colour=COLOURS.BACKGROUND
        )

        self._quit_button: Button = Button(
            x=button_x,
            y=480,
            width=button_width,
            height=button_height,
            text="Quit",
            colour=COLOURS.BACKGROUND
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

        # Dark background.
        self.screen.fill(COLOURS.BACKGROUND)

        # Title.
        draw_outlined_text(
            self.screen,
            "NEAT-ish Racing",
            (640, 150),
            align="center",
            font_size=64
        )

        # Buttons.
        self._train_button.draw(self.screen)
        self._play_button.draw(self.screen)
        self._quit_button.draw(self.screen)

        # Footer.
        draw_outlined_text(
            self.screen,
            "Press ESC to quit",
            (640, 650),
            align="center",
            font_size=18,
            text_colour=Color(100, 100, 100)
        )

        pygame.display.flip()
