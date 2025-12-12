from __future__ import annotations

import pygame

from pygame import Surface, Vector2
from pygame.time import Clock
from config import GAME, COLOURS
from src.algorithm import Genome
from src.training import AIController
from src.io import GenomeIO
from src.core import Car, Events, Track
from src.core.utils import draw_outlined_text
from .input_handler import InputHandler


class GameLoop:

    """
    Main game loop handling initialisation, updates, and rendering.

    Runs a fixed-timestep physics loop alongside variable-rate loops
    for non-physics updates and drawing. The player races against
    AI-controlled opponents loaded from genome files.

    Methods
    -------
    run() -> str | None
        Runs the game loop.
    """

    def __init__(self, track_path: str, genome_paths: list[str]) -> None:

        pygame.init()

        self._screen: Surface = pygame.display.set_mode((GAME.SCREEN_WIDTH, GAME.SCREEN_HEIGHT))
        self._clock: Clock = Clock()
        self._running: bool = True
        self._accumulator: float = 0.0

        pygame.display.set_caption("NEAT-ish Racing")

        # Shows a loading screen.
        self._screen.fill((0, 0, 0))
        draw_outlined_text(
            self._screen,
            "Loading...",
            (GAME.SCREEN_WIDTH // 2, GAME.SCREEN_HEIGHT // 2),
            text_colour=COLOURS.TEXT_SECONDARY
        )

        pygame.display.flip()

        # Loads the track - takes time!
        self._track: Track = Track(track_path)
        self._num_checkpoints: int = len(self._track.checkpoints)

        # Creates the player car at the player start position.
        self._player_car: Car = Car(start_pos=self._track.player_start_position)

        # Loads AI opponents from genome files.
        self._ai_controllers: list[AIController] = []
        self._load_ai_opponents(genome_paths)

    def _load_ai_opponents(self, genome_paths: list[str]) -> None:

        """
        Loads AI opponents from genome files.

        Parameters
        ----------
        genome_paths : list[str]
            A list of file paths to the genome files.
        """

        for i, path in enumerate(genome_paths):

            genome: Genome = GenomeIO.load_genome(path)
            start_pos: Vector2 = self._track.start_positions[i % len(self._track.start_positions)]
            car: Car = Car(start_pos=start_pos, colour=COLOURS.CARS[i])
            controller: AIController = AIController(car, genome)

            self._ai_controllers.append(controller)

    def run(self) -> str | None:

        """
        Runs the main game loop until the window is closed.

        Returns
        -------
        str | None
            'QUIT' if window was closed, ``None`` if ESC was pressed.
        """

        while self._running:

            result: str | None = self._process_events()

            # If X button was clicked, returns immediately.
            if result == 'QUIT':

                pygame.quit()
                return 'QUIT'

            # Calculates delta time and adds to the fixed accumulator.
            dt: float = self._clock.tick(GAME.FPS) / 1000.0
            self._accumulator += dt

            # Fixed timestep loop for deterministic physics.
            while self._accumulator >= GAME.FIXED_DT:

                self._fixed_update(GAME.FIXED_DT)
                self._accumulator -= GAME.FIXED_DT

            self._update(dt)
            self._draw()

        pygame.quit()
        return None

    def _process_events(self) -> str | None:

        """
        Processes all pending Pygame events.

        Returns
        -------
        str | None
            'QUIT' if X button clicked, ``None`` otherwise.
        """

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self._running = False
                return 'QUIT'

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return None

        return None

    def _update(self, dt: float) -> None:

        """
        Handles per-frame updates.

        Parameters
        ----------
        dt : float
            Time since the last frame, in seconds.
        """

        # Updates player sensors.
        self._player_car.update_sensors(self._track)

        # Updates AI sensors and decisions.
        for controller in self._ai_controllers:

            if not controller.is_alive:
                continue

            controller.car.update_sensors(self._track)
            controller.make_decision(dt)

    def _fixed_update(self, dt: float) -> None:

        """
        Handles fixed-timestep updates for physics and movement.

        Parameters
        ----------
        dt : float
            Fixed timestep duration, in seconds.
        """

        # Updates the player car.
        self._player_car.fixed_update(dt)
        self._handle_player_collisions()

        # Updates player input.
        InputHandler.fixed_update(self._player_car)

        # Updates AI cars.
        for controller in self._ai_controllers:

            if not controller.is_alive:
                continue

            controller.fixed_update()
            controller.car.fixed_update(dt)
            self._handle_ai_collisions(controller)

    def _handle_player_collisions(self) -> None:

        """
        Handles collision and checkpoint detection for the player car.
        """

        # Checks for collisions with the track bounds.
        if self._player_car.check_track_collision(self._track):
            Events.on_car_collided.broadcast(data=(self._player_car, self._track))

        checkpoint_order: int = self._track.check_checkpoint(
            self._player_car.position.x,
            self._player_car.position.y
        )

        # Checks for checkpoint crossing.
        if checkpoint_order >= 0:
            Events.on_checkpoint_hit.broadcast(
                data=(self._player_car, checkpoint_order, self._num_checkpoints)
            )

        # Checks for finish line crossing.
        if self._player_car.rect.colliderect(self._track.finish_line):
            Events.on_finish_line_crossed.broadcast(
                data=(self._player_car, self._num_checkpoints)
            )

    def _handle_ai_collisions(self, controller: AIController) -> None:

        """
        Handles collision and checkpoint detection for an AI controller.

        Parameters
        ----------
        controller : AIController
            The AI controller to check collisions for.
        """

        car = controller.car

        # Checks for collisions with the track bounds.
        if car.check_track_collision(self._track):
            controller.kill()
            return

        # Checks for checkpoint crossing.
        checkpoint_order: int = self._track.check_checkpoint(car.position.x, car.position.y)

        if checkpoint_order >= 0:
            controller.handle_checkpoint_hit(checkpoint_order, self._num_checkpoints)

        # Checks for finish line crossing.
        if car.rect.colliderect(self._track.finish_line):
            controller.handle_finish_line(self._num_checkpoints)

    def _draw(self) -> None:

        """
        Draws all the visible game elements on the screen.
        """

        # Draws the track.
        self._track.draw(self._screen)

        # Draws the player car.
        self._player_car.draw(self._screen)

        # Draws the AI cars.
        for controller in self._ai_controllers:
            controller.draw(self._screen)

        pygame.display.flip()
