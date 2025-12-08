import pygame

from pygame.surface import Surface
from pygame.time import Clock
from shapely.geometry import Point
from algorithm.config import GENETIC
from algorithm.ai_controller import AIController
from algorithm.genetic_algorithm import GeneticAlgorithm
from algorithm.genome import Genome
from game.config import CAR, GAME
from game.car import Car
from game.core.utils import draw_outlined_text
from game.events import Events
from game.track import Track


class TrainingLoop:

    def __init__(self) -> None:

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((1280, 720))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.accumulator: float = 0.0

        # Sets the window's title.
        pygame.display.set_caption("NEAT-ish Racing - Training Mode")

        # Loads the training track.
        self.track: Track = Track('./game/tracks/raw/track_0.tmx')

        # Initialises the genetic algorithm.
        self.genetic_algorithm: GeneticAlgorithm = GeneticAlgorithm(
            population_size=GENETIC.POPULATION_SIZE,
            input_size=len(CAR.SENSORS),
            output_size=4
        )

        self.controllers: list[AIController] = []
        self._generation_timer: float = 0.0
        self._create_generation()

        self._add_listeners()

    def _create_generation(self) -> None:

        """
        Creates a new generation of AI controllers.
        """

        # Disposes of any existing cars.
        for controller in self.controllers:
            controller.car.dispose()

        self.controllers = []
        self.generation_timer = 0.0

        for genome, _ in self.genetic_algorithm.population:

            car: Car = Car(start_pos=self.track.start_pos)
            controller: AIController = AIController(car, genome)
            self.controllers.append(controller)

    def _add_listeners(self) -> None:

        """
        Adds methods as event listeners.
        """

        Events.on_car_moved.add_listener(self._check_collisions)
        Events.on_car_moved.add_listener(self._check_checkpoints)

    def run(self) -> None:

        """
        Runs the main training loop until the window is closed.
        """

        while self.running:

            self._process_events()

            dt: float = self.clock.tick(GAME.FPS) / 1000.0
            self.accumulator += dt
            self.generation_timer += dt

            # Fixed timestep updates
            while self.accumulator >= GAME.FIXED_DT:

                self._fixed_update(GAME.FIXED_DT)
                self.accumulator -= GAME.FIXED_DT

            self._update(dt)
            self._draw()

            # Check if generation is complete
            if self._is_generation_complete():
                self._next_generation()

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

        for controller in self.controllers:

            controller.update(dt)

            if controller.car.is_alive:
                controller.car.update_sensors(self.track)

    def _fixed_update(self, dt: float) -> None:

        """
        Handles fixed-timestep updates for physics and movement.

        Parameters
        ----------
        dt : float
            Fixed timestep duration, in seconds.
        """

        # Updates only the cars which are alive.
        for controller in (c for c in self.controllers if c.car.is_alive):
            controller.car.fixed_update(dt)

    def _draw(self) -> None:

        """
        Draws all the visible game elements on the screen.
        """

        self.track.draw(self.screen)

        for controller in self.controllers:
            controller.car.draw(self.screen)

        # Draw stats
        self._draw_stats()

        pygame.display.flip()

    def _draw_stats(self) -> None:

        """
        Draws training statistics on screen.
        """

        alive_count = sum(c.car.is_alive for c in self.controllers)
        time_remaining = max(0, GENETIC.MAX_GENERATION_TIME - self.generation_timer)
        best_fitness = max(c.calculate_fitness() for c in self.controllers)

        draw_outlined_text(self.screen, f"Generation: {self.genetic_algorithm.generation}", (10, 10), align="left")
        draw_outlined_text(self.screen, f"Alive: {alive_count}/{GENETIC.POPULATION_SIZE}", (10, 35), align="left")
        draw_outlined_text(self.screen, f"Time: {time_remaining:.1f}s", (10, 60), align="left")
        draw_outlined_text(self.screen, f"Best Fitness: {best_fitness:.0f}", (10, 85), align="left")

    def _is_generation_complete(self) -> bool:

        """
        Checks if the current generation is complete.

        Returns
        -------
        bool
            ``True`` if generation should end, ``False`` otherwise.
        """

        # Checks if all cars are dead.
        all_dead = all(not c.car.is_alive for c in self.controllers)

        # Checks if the time limit has been reached.
        time_up = self.generation_timer >= GENETIC.MAX_GENERATION_TIME

        return all_dead or time_up

    def _next_generation(self) -> None:

        """
        Evolves to the next generation.
        """

        # Maps genomes to controllers.
        genome_to_controller = {c.genome: c for c in self.controllers}

        # Calculates the fitness for all controllers.
        def fitness_func(genome: Genome) -> float:
            c = genome_to_controller.get(genome)
            return c.calculate_fitness() if c else 0.0

        # Evolves the population.
        self.genetic_algorithm.next_generation(fitness_func)

        # Gets the average fitness scores.
        average_fitness = sum(c.fitness for c in self.controllers) / len(self.controllers)

        print(f"""
        === Generation {self.genetic_algorithm.generation - 1} Complete ===
        Average Fitness: {average_fitness:.2f}
        """)

        # Creates a new generation.
        self._create_generation()

    def _check_collisions(self, data) -> None:

        car, shape_points = data

        for point in shape_points:

            if not self.track.is_on_track(point):
                Events.on_car_collided.broadcast(data=(car, self.track))
                return

    def _check_checkpoints(self, data) -> None:

        car, shape_points = data
        car_point = Point(car.position.x, car.position.y)

        # Checks for checkpoint collisions.
        for checkpoint in self.track.checkpoints:
            if checkpoint.shape.contains(car_point):
                Events.on_checkpoint_hit.broadcast(data=(car, checkpoint.order))

        # Checks for finish line collisions.
        if car.rect.colliderect(self.track.finish_line):
            Events.on_finish_line_crossed.broadcast(data=(car, len(self.track.checkpoints)))
