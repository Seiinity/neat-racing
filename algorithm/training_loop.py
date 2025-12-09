import pygame
from pygame import Vector2

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

    def __init__(self, genome: Genome | None = None) -> None:

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
            output_size=4,
            base_genome=genome
        )

        self.controllers: list[AIController] = []
        self._generation_timer: float = 0.0
        self._physics_step_count: int = 0  # Counts physics steps for AI synchronization
        self._create_generation()

        self._add_listeners()

        # Force render.
        Events.on_keypress_checkpoints.broadcast()

    def _create_generation(self) -> None:

        """
        Creates a new generation of AI controllers.
        """

        # Disposes of any existing controllers.
        for controller in self.controllers:
            controller.dispose()

        self.controllers = []
        self._generation_timer = 0.0
        self._physics_step_count = 0  # Reset step counter

        for i, (genome, _) in enumerate(self.genetic_algorithm.population):

            start_pos: Vector2 = self.track.start_positions[i % len(self.track.start_positions)]
            car: Car = Car(start_pos=start_pos)
            controller: AIController = AIController(car, genome)
            self.controllers.append(controller)

        # Force render.
        Events.on_keypress_sensors.broadcast()

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

            # Fixed timestep loop - everything runs here for determinism
            while self.accumulator >= GAME.FIXED_DT:
                self._fixed_update(GAME.FIXED_DT)
                self._generation_timer += GAME.FIXED_DT  # Timer increments with fixed steps only!
                self.accumulator -= GAME.FIXED_DT

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

    def _fixed_update(self, dt: float) -> None:

        """
        Handles fixed-timestep updates for physics and AI.

        Parameters
        ----------
        dt : float
            Fixed timestep duration, in seconds.

        Notes
        -----
        The AI controller only makes decisions every ``AI_STEPS_INTERVAL``
        physics steps for performance, while maintaining determinism.
        """

        # Checks whether the AI controller should make a decision this physics step.
        run_ai = (self._physics_step_count % GENETIC.TRAINING_INTERVAL == 0)

        # Updates all living controllers and cars.
        for controller in [c for c in self.controllers if c.is_alive]:

            # Waits for actions from the AI controller, sparsely.
            if run_ai:
                controller.car.update_sensors(self.track)
                controller.make_decision(dt * GENETIC.TRAINING_INTERVAL)

            # Applies the current actions and updates the car.
            controller.fixed_update()
            controller.car.fixed_update(dt)

        self._physics_step_count += 1

    def _draw(self) -> None:

        """
        Draws all the visible game elements on the screen.
        """

        self.track.draw(self.screen)

        # Calculate fitness for all controllers
        fitness_values = [(c, c.fitness) for c in self.controllers]

        # Find best and worst fitness values
        best_fitness = max(fitness_values, key=lambda x: x[1])[1]
        worst_fitness = min(fitness_values, key=lambda x: x[1])[1]

        for controller in self.controllers:
            controller_fitness = next(f for c, f in fitness_values if c == controller)
            is_best = (controller_fitness == best_fitness)
            is_worst = (controller_fitness == worst_fitness) and not is_best
            controller.draw(self.screen, is_best, is_worst)

        # Draw stats
        self._draw_stats()

        pygame.display.flip()

    def _draw_stats(self) -> None:

        """
        Draws training statistics on screen.
        """

        alive_count = sum(c.is_alive for c in self.controllers)
        time_remaining = max(0, GENETIC.MAX_GENERATION_TIME - self._generation_timer)
        best_fitness = max(c.fitness for c in self.controllers)

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
        all_dead = all(not c.is_alive for c in self.controllers)

        # Checks if the time limit has been reached.
        time_up = self._generation_timer >= GENETIC.MAX_GENERATION_TIME

        return all_dead or time_up

    def _next_generation(self) -> None:

        """
        Evolves to the next generation.
        """

        # Maps genomes to controllers.
        genome_to_controller = {id(c.genome): c for c in self.controllers}

        # Calculates the fitness for all controllers.
        def fitness_func(genome: Genome) -> float:
            c = genome_to_controller.get(id(genome))
            return c.fitness if c else 0.0

        # Evolves the population.
        self.genetic_algorithm.next_generation(fitness_func)

        # Gets the average fitness scores.
        average_fitness = sum(c.fitness for c in self.controllers) / len(self.controllers)

        print(f"""
        === Generation {self.genetic_algorithm.generation - 1} Complete ===
        Average Fitness: {average_fitness:.2f}
        Elite genome weights sum: {self.controllers[0].genome.weights.sum():.6f}
        """)

        # Creates a new generation.
        self._create_generation()

    def _check_collisions(self, data) -> None:

        car, shape_points = data

        for point in shape_points:

            if not self.track.is_on_track(point):
                Events.on_car_die.broadcast(data=car)
                return

    def _check_checkpoints(self, data) -> None:

        car, shape_points = data
        car_point = Point(car.position.x, car.position.y)

        # Checks for checkpoint collisions.
        for checkpoint in self.track.checkpoints:
            if checkpoint.shape.contains(car_point):
                Events.on_checkpoint_hit.broadcast(data=(car, checkpoint.order, len(self.track.checkpoints)))

        # Checks for finish line collisions.
        if car.rect.colliderect(self.track.finish_line):
            Events.on_finish_line_crossed.broadcast(data=(car, len(self.track.checkpoints)))
