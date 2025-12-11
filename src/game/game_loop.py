from __future__ import annotations

import pygame

from pygame import Surface
from pygame.time import Clock
from config import GAME
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
    player_car : Car
        The player-controlled car.
    ai_controllers : list[AIController]
        A list of AI controllers for the opponent cars.
    """

    def __init__(self, track_path: str, genome_paths: list[str]) -> None:

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((1280, 720))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.accumulator: float = 0.0

        pygame.display.set_caption("NEAT-ish Racing")

        # Shows a loading screen.
        self.screen.fill((0, 0, 0))
        draw_outlined_text(
            self.screen,
            "Loading...",
            (GAME.SCREEN_WIDTH // 2, GAME.SCREEN_HEIGHT // 2)
        )

        pygame.display.flip()

        # Loads the track.
        self.track: Track = Track(track_path)

        # Caches the number of checkpoints to avoid repeated lookups.
        self._num_checkpoints: int = len(self.track.checkpoints)

        # Creates the player car at the player start position.
        self.player_car: Car = Car(start_pos=self.track.player_start_position)

        # Loads AI opponents from genome files.
        self.ai_controllers: list[AIController] = []
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

            genome = GenomeIO.load_genome(path)
            start_pos = self.track.start_positions[i % len(self.track.start_positions)]
            car = Car(start_pos=start_pos)
            controller = AIController(car, genome)

            self.ai_controllers.append(controller)

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

        # Updates player sensors.
        self.player_car.update_sensors(self.track)

        # Updates AI sensors and decisions.
        for controller in self.ai_controllers:

            if not controller.is_alive:
                continue

            controller.car.update_sensors(self.track)
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
        self.player_car.fixed_update(dt)
        self._handle_player_collisions()

        # Updates AI cars.
        for controller in self.ai_controllers:

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
        if self.player_car.check_track_collision(self.track):
            Events.on_car_collided.broadcast(data=(self.player_car, self.track))

        # Checks for checkpoint crossing.
        checkpoint_order: int = self.track.check_checkpoint(
            self.player_car.position.x,
            self.player_car.position.y
        )

        if checkpoint_order >= 0:
            Events.on_checkpoint_hit.broadcast(
                data=(self.player_car, checkpoint_order, self._num_checkpoints)
            )

        # Checks for finish line crossing.
        if self.player_car.rect.colliderect(self.track.finish_line):
            Events.on_finish_line_crossed.broadcast(
                data=(self.player_car, self._num_checkpoints)
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
        if car.check_track_collision(self.track):
            controller.kill()
            return

        # Checks for checkpoint crossing.
        checkpoint_order: int = self.track.check_checkpoint(car.position.x, car.position.y)

        if checkpoint_order >= 0:
            controller.handle_checkpoint_hit(checkpoint_order, self._num_checkpoints)

        # Checks for finish line crossing.
        if car.rect.colliderect(self.track.finish_line):
            controller.handle_finish_line(self._num_checkpoints)

    def _draw(self) -> None:

        """
        Draws all the visible game elements on the screen.
        """

        self.track.draw(self.screen)

        # Draws the player car in a distinct colour.
        self.player_car.draw(self.screen, pygame.Color(0, 200, 255))

        # Draws AI cars.
        for controller in self.ai_controllers:
            controller.draw(self.screen)

        pygame.display.flip()