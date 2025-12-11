import pygame

from pathlib import Path
from pygame import Vector2, Surface
from pygame.time import Clock
from config import COLOURS, TRAINING, CAR, GAME
from src.algorithm import GeneticAlgorithm, Genome
from src.io import GenomeIO
from src.core.car import Car, Track
from src.core.utils import draw_outlined_text
from src.ui import Button
from .ai_controller import AIController


class TrainingLoop:

    """
    Training mode with toggleable visual and console modes.

    Attributes
    ----------
    screen : Surface
        The Pygame display surface.
    clock : Clock
        The Pygame clock for timing.
    running : bool
        Whether the training loop is running.
    visual_mode : bool
        Whether visual mode is active (False = console mode).
    track : Track
        The racing track.
    genetic_algorithm : GeneticAlgorithm
        The genetic algorithm managing the population.
    controllers : list[AIController]
        List of AI controllers for the current generation.
    best_genome_ever : Genome | None
        The best genome found across all generations.
    best_fitness_ever : float
        The best fitness score achieved across all generations.

    Methods
    -------
    run(self) -> None
        Runs the main training loop.
    """

    def __init__(
            self,
            track_path: str,
            genome: Genome | None = None,
            save_interval: int = 10,
            save_best: bool = True
    ) -> None:

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((GAME.SCREEN_WIDTH, GAME.SCREEN_HEIGHT))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.accumulator: float = 0.0

        pygame.display.set_caption("NEAT-ish Racing - Training")

        # Starts in console mode for maximum performance.
        self.visual_mode: bool = False

        # Creates the toggle button.
        self.toggle_button = Button(
            x=(GAME.SCREEN_WIDTH // 2) - 100,
            y=GAME.SCREEN_HEIGHT // 2,
            width=200,
            height=40,
            text="Show Training"
        )

        # Genome saving configuration.
        self.save_interval: int = save_interval
        self.save_best: bool = save_best
        self.save_dir = Path("./data/genomes")
        self.save_dir.mkdir(exist_ok=True)
        self.best_genome_ever: Genome | None = None
        self.best_fitness_ever: float = -float('inf')

        # Shows a loading screen.
        self.screen.fill((0, 0, 0))
        draw_outlined_text(
            self.screen,
            "Loading...",
            (GAME.SCREEN_WIDTH // 2, GAME.SCREEN_HEIGHT // 2),
            text_colour=COLOURS.TEXT_SECONDARY
        )

        pygame.display.flip()

        # Loads the track.
        self.track: Track = Track(track_path)
        self._num_checkpoints: int = len(self.track.checkpoints)

        # Initialises the genetic algorithm.
        self.genetic_algorithm: GeneticAlgorithm = GeneticAlgorithm(
            population_size=TRAINING.POPULATION_SIZE,
            input_size=len(CAR.SENSORS),
            output_size=4,
            base_genome=genome
        )

        self.controllers: list[AIController] = []
        self._generation_timer: float = 0.0
        self._physics_step_count: int = 0
        self._total_generations: int = 0
        self._current_speed: int = TRAINING.SPEED

        # For periodic console updates.
        self._status_update_interval: float = 5.0
        self._last_status_time: float = 0.0

        # Stats history for tracking progress.
        self._fitness_history: list[tuple[float, float, float]] = []

        self._print_startup_banner()
        self._create_generation()

    def _print_startup_banner(self) -> None:

        """
        Prints startup information to the console.
        """

        print("=" * 60)
        print("NEAT-ish Racing - Training Mode")
        print("=" * 60)
        print(f"Track loaded: {self._num_checkpoints} checkpoints")
        print(f"Population: {TRAINING.POPULATION_SIZE} cars")
        print(f"Training speed: {self._current_speed}x")
        print(f"Save directory: {self.save_dir.absolute()}")
        print("=" * 60)

    def _create_generation(self) -> None:

        """
        Creates a new generation of AI controllers.
        """

        # Disposes of any existing controllers.
        for controller in self.controllers:
            controller.dispose()

        self.controllers = []
        self._generation_timer = 0.0
        self._physics_step_count = 0
        self._last_status_time = 0.0

        # Creates a controller for each genome in the population.
        for i, (genome, _) in enumerate(self.genetic_algorithm.population):
            start_pos: Vector2 = self.track.start_positions[i % len(self.track.start_positions)]
            car: Car = Car(start_pos=start_pos)
            controller: AIController = AIController(car, genome)
            self.controllers.append(controller)

        print(f"\nGeneration {self.genetic_algorithm.generation} started.")

    def run(self) -> None:

        """
        Runs the main training loop.

        Notes
        -----
        The loop can be interrupted with Ctrl+C to gracefully stop training
        and print final statistics.
        """

        try:

            while self.running:

                self._process_events()
                self._current_speed = 1 if self.visual_mode else TRAINING.SPEED

                dt: float = self.clock.tick(GAME.FPS) / 1000.0
                self.accumulator += dt * self._current_speed

                # Fixed timestep loop for deterministic physics.
                while self.accumulator >= GAME.FIXED_DT:
                    self._fixed_update(GAME.FIXED_DT)
                    self._generation_timer += GAME.FIXED_DT
                    self.accumulator -= GAME.FIXED_DT

                # Prints periodic status updates in console mode.
                if not self.visual_mode:
                    if self._generation_timer - self._last_status_time >= self._status_update_interval:
                        self._print_console_status()
                        self._last_status_time = self._generation_timer

                # Renders based on current mode.
                if self.visual_mode:
                    self._draw_visual()
                else:
                    self._draw_minimal_gui()

                # Checks if generation is complete.
                if self._is_generation_complete():
                    self._next_generation()

        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("Training interrupted.")
            self._print_final_stats()

        pygame.quit()

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
                elif event.key == pygame.K_SPACE:
                    self._toggle_mode()

            # Handles button clicks.
            if self.toggle_button.handle_event(event):
                self._toggle_mode()

    def _toggle_mode(self) -> None:

        """
        Toggles between visual and console mode.

        Notes
        -----
        When switching to visual mode, the window is resized to 1280x720.
        When switching to console mode, the window is resized to 400x100.
        """

        self.visual_mode = not self.visual_mode

        if self.visual_mode:

            # Switches to large window for visual mode.
            self.screen = pygame.display.set_mode((1280, 720))
            self.toggle_button.rect.topleft = (1280 - 220, 20)
            self.toggle_button.text = "Hide Training"

        else:

            # Switches back to small window for console mode.
            self.screen = pygame.display.set_mode((400, 100))
            self.toggle_button.rect.topleft = (125, 50)
            self.toggle_button.text = "Show Training"

    def _fixed_update(self, dt: float) -> None:

        """
        Handles fixed-timestep updates for physics and AI.

        Parameters
        ----------
        dt : float
            Fixed timestep duration, in seconds.

        Notes
        -----
        The AI controller only makes decisions every ``TRAINING_INTERVAL``
        physics steps for performance, while maintaining determinism.
        """

        # Checks whether the AI controller should make a decision this physics step.
        run_ai: bool = (self._physics_step_count % TRAINING.INTERVAL == 0)

        # Updates all living controllers and cars.
        for controller in self.controllers:

            if not controller.is_alive:
                continue

            if run_ai:
                controller.car.update_sensors(self.track)
                controller.make_decision(dt * TRAINING.INTERVAL)

            controller.fixed_update()
            controller.car.fixed_update(dt)

            # Handles collisions directly without event system.
            self._handle_collisions(controller)

        self._physics_step_count += 1

    def _handle_collisions(self, controller: AIController) -> None:

        """
        Handles collision and checkpoint detection for a single controller.

        Parameters
        ----------
        controller : AIController
            The AI controller to check collisions for.

        Notes
        -----
        Uses direct method calls instead of the event system for performance.
        Uses the track's fast collision mask for O(1) point-in-track checks.
        """

        car = controller.car

        # Checks for collisions with the track bounds using the fast method.
        if car.check_track_collision(self.track):
            controller.kill()
            return

        # Checks for checkpoint crossing using fast bounding box pre-rejection.
        checkpoint_order: int = self.track.check_checkpoint(car.position.x, car.position.y)

        if checkpoint_order >= 0:
            controller.handle_checkpoint_hit(checkpoint_order, self._num_checkpoints)

        # Checks for finish line crossing.
        if car.rect.colliderect(self.track.finish_line):
            controller.handle_finish_line(self._num_checkpoints)

    def _print_console_status(self) -> None:

        """
        Prints current generation status to the console.
        """

        alive_count = sum(c.is_alive for c in self.controllers)
        time_remaining = max(0, TRAINING.MAX_GENERATION_TIME - self._generation_timer)

        if not self.controllers:
            return

        fitness_values = [c.fitness for c in self.controllers]
        best_fitness = max(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)

        best_controller = max(self.controllers, key=lambda c: c.fitness)
        best_car = best_controller.car

        print(
            f"  [{self._generation_timer:.1f}s] "
            f"Alive: {alive_count:2d}/{TRAINING.POPULATION_SIZE} | "
            f"Best: {best_fitness:7.0f} | "
            f"Avg: {avg_fitness:7.0f} | "
            f"CP: {best_car.current_checkpoint}/{self._num_checkpoints} | "
            f"Laps: {best_car.laps_completed} | "
            f"Time left: {time_remaining:.1f}s"
        )

    def _draw_minimal_gui(self) -> None:

        """
        Draws minimal GUI with just the toggle button.

        Notes
        -----
        This is used in console mode to keep the window small and unobtrusive
        while still allowing the user to toggle to visual mode.
        """

        # Dark background.
        self.screen.fill(COLOURS.BACKGROUND)

        # Title text.
        draw_outlined_text(
            self.screen,
            "Training Mode: Console Output",
            (GAME.SCREEN_WIDTH // 2, GAME.SCREEN_HEIGHT // 2 - 20),
            text_colour=COLOURS.TEXT_SECONDARY
        )

        # Toggle button.
        self.toggle_button.draw(self.screen)

        pygame.display.flip()

    def _draw_visual(self) -> None:

        """
        Draws the full visual mode with track and cars.

        Notes
        -----
        This mode draws the track background, all cars, stats overlay,
        and the toggle button.
        """

        # Draws the track.
        self.track.draw(self.screen)

        # Draws cars with colour coding.
        if self.controllers:

            fitness_values = [c.fitness for c in self.controllers]
            best_fitness = max(fitness_values)
            worst_fitness = min(fitness_values)

            for controller in self.controllers:
                is_best = (controller.fitness == best_fitness)
                is_worst = (controller.fitness == worst_fitness) and not is_best
                controller.draw(self.screen, is_best, is_worst)

        # Draws stats overlay.
        self._draw_visual_stats_overlay()

        # Draws toggle button.
        self.toggle_button.draw(self.screen)

        pygame.display.flip()

    def _draw_visual_stats_overlay(self) -> None:

        """
        Draws stats overlay in visual mode.

        Notes
        -----
        Displays the generation number, cars alive, time remaining,
        the best fitness, average fitness, and best car's progress.
        """

        alive_count = sum(c.is_alive for c in self.controllers)
        time_remaining = max(0, TRAINING.MAX_GENERATION_TIME - self._generation_timer)

        if self.controllers:
            fitness_values = [c.fitness for c in self.controllers]
            best_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)

            best_controller = max(self.controllers, key=lambda c: c.fitness)
            best_car = best_controller.car
        else:
            best_fitness = avg_fitness = 0
            best_car = None

        # Semi-transparent background.
        overlay = pygame.Surface((350, 180))
        overlay.set_alpha(200)
        overlay.fill(COLOURS.ITEM_UNSELECTED)
        self.screen.blit(overlay, (10, 10))

        # Stats.
        y = 20
        draw_outlined_text(
            self.screen, f"Generation: {self.genetic_algorithm.generation}",
            (20, y), align="left", font_size=20
        )

        y += 30
        draw_outlined_text(
            self.screen, f"Alive: {alive_count}/{TRAINING.POPULATION_SIZE}",
            (20, y), align="left", font_size=20
        )

        y += 30
        draw_outlined_text(
            self.screen, f"Time: {time_remaining:.1f}s / {TRAINING.MAX_GENERATION_TIME:.0f}s",
            (20, y), align="left", font_size=20
        )

        y += 30
        draw_outlined_text(
            self.screen, f"Best Fitness: {best_fitness:.0f}",
            (20, y), align="left", font_size=20
        )

        y += 30
        draw_outlined_text(
            self.screen, f"Avg Fitness: {avg_fitness:.0f}",
            (20, y), align="left", font_size=20
        )

        if best_car:
            y += 30
            draw_outlined_text(
                self.screen,
                f"Best CP: {best_car.current_checkpoint}/{self._num_checkpoints} | "
                f"Laps: {best_car.laps_completed}",
                (20, y), align="left", font_size=18
            )

    def _is_generation_complete(self) -> bool:

        """
        Checks if the current generation is complete.

        Returns
        -------
        bool
            ``True`` if the generation should end, ``False`` otherwise.

        Notes
        -----
        A generation is complete when either all cars are dead or the
        maximum generation time has been reached.
        """

        all_dead = all(not c.is_alive for c in self.controllers)
        time_up = self._generation_timer >= TRAINING.MAX_GENERATION_TIME

        return all_dead or time_up

    def _next_generation(self) -> None:

        """
        Evolves to the next generation and saves genomes.

        Notes
        -----
        This method calculates fitness statistics, evolves the population,
        saves the best genomes, and creates the next generation.
        """

        genome_to_controller = {id(c.genome): c for c in self.controllers}

        def fitness_func(g: Genome) -> float:
            c = genome_to_controller.get(id(g))
            return c.fitness if c else 0.0

        # Evolves the population.
        self.genetic_algorithm.next_generation(fitness_func)

        # Calculates statistics.
        fitness_values = [c.fitness for c in self.controllers]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)

        # Stores history.
        self._fitness_history.append((avg_fitness, max_fitness, min_fitness))
        self._total_generations += 1

        best_controller = max(self.controllers, key=lambda c: c.fitness)
        best_car = best_controller.car

        gen_num = self.genetic_algorithm.generation - 1

        # Console output for generation completion.
        print(f"\n{'=' * 60}")
        print(f"Generation {gen_num} Complete")
        print(f"{'=' * 60}")
        print(f"  Average Fitness:  {avg_fitness:10.2f}")
        print(f"  Best Fitness:     {max_fitness:10.2f}")
        print(f"  Worst Fitness:    {min_fitness:10.2f}")
        print(f"  Best Checkpoints: {best_car.current_checkpoint}/{self._num_checkpoints}")
        print(f"  Best Laps:        {best_car.laps_completed}")
        print(f"  Generation Time:  {self._generation_timer:.2f}s")

        # Saves best genome ever.
        if self.save_best and max_fitness > self.best_fitness_ever:

            self.best_fitness_ever = max_fitness
            self.best_genome_ever = self.genetic_algorithm.get_top(1)[0].copy()

            filename = self.save_dir / f"best_gen_{gen_num}_fitness_{max_fitness:.0f}.pkl"
            GenomeIO.save_genome(self.best_genome_ever, str(filename))

            print(f"  New best genome saved: {filename.name}")

        # Periodic autosave.
        if gen_num % self.save_interval == 0 and gen_num > 0:

            top_genomes = self.genetic_algorithm.get_top(3)

            for i, genome in enumerate(top_genomes):
                filename = self.save_dir / f"gen_{gen_num}_rank_{i + 1}.pkl"
                GenomeIO.save_genome(genome, str(filename))

            print(f"  Autosaved top 3 genomes for generation {gen_num}.")

        print(f"{'=' * 60}")

        # Creates a new generation.
        self._create_generation()

    def _print_final_stats(self) -> None:

        """
        Prints final statistics when training stops.
        """

        print(f"Total Generations Completed: {self._total_generations}")

        if self.best_genome_ever:
            print(f"Best Fitness Ever Achieved: {self.best_fitness_ever:.2f}")

        print("=" * 60)
