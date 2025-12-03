import pygame

from pygame import Surface
from pygame.time import Clock
from game.car import Car
from game.input_handler import InputHandler
from game.config import GAME

class GameLoop:

    """
    Main game loop handling initialisation, updates, and rendering.

    Runs a fixed-timestep physics loop alongside variable-rate loops
    for non-physics updates and drawing.

    Attributes
    ----------
    screen : Surface
        The main screen for rendering.
    clock : Clock
        A clock used to manage frame rate.
    running : bool
        Whether the game loop is currently active.
    accumulator : float
        A time accumulator for fixed-timestep updates.
    cars : list[Car]
        A list of cars in the game.
    """

    def __init__(self) -> None:

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((1280, 720))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.accumulator: float = 0.0

        # Sets the window's title.
        pygame.display.set_caption('NEAT-ish Racing')

        # TODO: Instantiate multiple cars.
        self.cars: list[Car] = [Car()]

    def run(self) -> None:

        """
        Runs the main game loop until the window is closed.
        """

        while self.running:

            self._process_events()

            dt: float = self.clock.tick(GAME.FPS) / 1000.0
            self.accumulator += dt

            while self.accumulator >= GAME.FIXED_DT:

                self._fixed_update(GAME.FIXED_DT)
                self.accumulator -= GAME.FIXED_DT

            self._update(dt)
            self._draw()

        pygame.quit()

    def _process_events(self) -> None:

        """
        Processes all pending Pygame events.
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _update(self, dt: float) -> None:

        """
        Handles per-frame updates.

        Parameters
        ----------
        dt : float
            Time since the last frame, in seconds.
        """

        InputHandler.update()

    def _fixed_update(self, dt: float) -> None:

        """
        Handles fixed-timestep updates for physics and movement.

        Parameters
        ----------
        dt : float
            Fixed timestep duration, in seconds.
        """

        for car in self.cars:
            car.fixed_update(dt)

    def _draw(self) -> None:

        """
        Draws all the visible game elements on the screen.
        """

        self.screen.fill("black")

        for  car in self.cars:
            car.draw(self.screen)

        pygame.display.flip()