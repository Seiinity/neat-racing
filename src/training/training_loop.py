import pygame
import multiprocessing as mp

from typing import Any
from pathlib import Path
from queue import Empty
from pygame import Vector2, Surface
from pygame.time import Clock
from config import COLOURS, FONTS, TRAINING, GAME, CONTROLLER
from src.algorithm import GeneticAlgorithm, Genome
from src.io import GenomeIO
from src.core.car import Car, Track
from src.core.utils import draw_outlined_text
from src.ui import Button, plotting_process
from .ai_controller import AIController
from ..core import Events


class TrainingLoop:

    """
    Training mode with toggleable visual and console modes.

    Attributes
    ----------
    genetic_algorithm : GeneticAlgorithm
        The genetic algorithm managing the population.

    Methods
    -------
    run() -> None
        Runs the main training loop.
    """

    def __init__(self, track_path: str) -> None:

        # Initialises the genetic algorithm.
        self.genetic_algorithm: GeneticAlgorithm = GeneticAlgorithm(
            population_size=TRAINING.POPULATION_SIZE,
            input_size=len(CONTROLLER.SENSORS),
            output_size=4
        )

        pygame.init()

        self._screen: Surface = pygame.display.set_mode((GAME.SCREEN_WIDTH, GAME.SCREEN_HEIGHT))
        self._clock: Clock = Clock()
        self._running: bool = True
        self._accumulator: float = 0.0

        pygame.display.set_caption("NEAT-ish Racing - Training")

        # Starts in console mode for maximum performance.
        self._visual_mode: bool = False

        # Creates the plot-displaying process.
        self._plot_queue: mp.Queue | None = None
        self._plot_process: mp.Process | None = None

        # Creates the graph button.
        self._graph_button: Button = Button(
            x=(GAME.SCREEN_WIDTH // 2) - 100,
            y=GAME.SCREEN_HEIGHT // 2 + 70,
            width=200,
            height=40,
            text="Show Graphs"
        )

        # Creates the toggle button.
        self._toggle_button: Button = Button(
            x=(GAME.SCREEN_WIDTH // 2) - 100,
            y=GAME.SCREEN_HEIGHT // 2 - 20,
            width=200,
            height=40,
            text="Show Training"
        )

        # Creates the stop button.
        self._stop_button: Button = Button(
            x=(GAME.SCREEN_WIDTH // 2) - 100,
            y=GAME.SCREEN_HEIGHT // 2 + 25,
            width=200,
            height=40,
            text="Stop Training"
        )

        # Genome saving configuration.
        self._save_interval: int = TRAINING.AUTOSAVE_INTERVAL
        self._save_dir: Path = Path("./data/genomes")
        self._save_dir.mkdir(exist_ok=True)

        # Shows a loading screen.
        self._screen.fill(COLOURS.BACKGROUND)
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

        self._controllers: list[AIController] = []
        self._generation_timer: float = 0.0
        self._physics_step_count: int = 0
        self._total_generations: int = 0
        self._current_speed: int = TRAINING.SPEED

        # For periodic console updates.
        self._status_update_interval: float = 5.0
        self._last_status_time: float = 0.0

        # Saves stats for display with seaborn.
        self._plot_background_sent: bool = False
        self._plot_history: dict[str, list[Any]] = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'fitness_std': [],
            'death_positions': []
        }

        # Force renders the checkpoints.
        Events.on_keypress_checkpoints.broadcast()

        self._print_startup()
        self._create_generation()

    def _print_startup(self) -> None:

        """
        Prints startup information to the console.
        """

        print("=" * 60)
        print("NEAT-ish Racing - Training Mode")
        print("=" * 60)
        print(f"Track loaded: {self._num_checkpoints} checkpoints")
        print(f"Population: {TRAINING.POPULATION_SIZE} cars")
        print(f"Training speed: {self._current_speed}x")
        print(f"Save directory: {self._save_dir.absolute()}")
        print("=" * 60)

    def _create_generation(self) -> None:

        """
        Creates a new generation of AI controllers.
        """

        # Disposes of any existing controllers.
        for controller in self._controllers:
            controller.dispose()

        self._controllers = []
        self._generation_timer = 0.0
        self._physics_step_count = 0
        self._last_status_time = 0.0

        # Creates a controller for each genome in the population.
        for i, (genome, _) in enumerate(self.genetic_algorithm.population):

            start_pos: Vector2 = self._track.start_positions[i % len(self._track.start_positions)]
            car: Car = Car(start_pos=start_pos)
            controller: AIController = AIController(car, genome)
            self._controllers.append(controller)

        print(f"\nGeneration {self.genetic_algorithm.generation} started.")

    def run(self) -> str | None:

        """
        Runs the main training loop.

        Returns
        -------
        str | None
            'QUIT' if window was closed, ``None`` if ESC was pressed.
        """

        try:

            while self._running:

                result = self._process_events()

                # If X button was clicked, returns immediately.
                if result == 'QUIT':

                    pygame.quit()
                    return 'QUIT'

                # Runs at normal speed if visualising the training.
                # Otherwise, runs at whatever training speed is specified in the config.
                self._current_speed = 1 if self._visual_mode else TRAINING.SPEED

                # Calculates delta time and adds to the fixed accumulator.
                dt: float = self._clock.tick(GAME.FPS) / 1000.0
                self._accumulator += dt * self._current_speed

                # Caps physics steps per frame to keep UI responsive.
                steps_this_frame = 0
                max_steps = 50

                # Fixed timestep loop for deterministic physics.
                while self._accumulator >= GAME.FIXED_DT and steps_this_frame < max_steps:

                    self._fixed_update(GAME.FIXED_DT)
                    self._generation_timer += GAME.FIXED_DT
                    self._accumulator -= GAME.FIXED_DT
                    steps_this_frame += 1

                # Resets the accumulator.
                if self._accumulator > GAME.FIXED_DT * max_steps:
                    self._accumulator = 0.0

                # Prints periodic status updates in console mode.
                if not self._visual_mode:
                    if self._generation_timer - self._last_status_time >= self._status_update_interval:
                        self._print_console_status()
                        self._last_status_time = self._generation_timer

                # Renders based on current mode.
                if self._visual_mode:
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

        # Cleans up plotting process.
        if self._plot_queue is not None:
            self._plot_queue.put(None)
        if self._plot_process is not None:
            self._plot_process.join(timeout=1)

        pygame.quit()
        return None

    def _process_events(self) -> str | None:

        """
        Processes all pending Pygame events.

        Returns
        -------
        str | None
            'QUIT' if X button clicked, None otherwise.
        """

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self._running = False
                return 'QUIT'

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return None

                elif event.key == pygame.K_SPACE:
                    self._toggle_mode()

            if self._toggle_button.handle_event(event):
                self._toggle_mode()

            if self._stop_button.handle_event(event):
                self._running = False
                return None

            if self._graph_button.handle_event(event):
                self._toggle_graphs()

        return None

    def _toggle_graphs(self) -> None:

        """
        Opens or closes the graph window.
        """

        if self._plot_process is None or not self._plot_process.is_alive():

            # Starts the plotting process.
            self._plot_queue = mp.Queue()
            self._plot_process = mp.Process(target=plotting_process, args=(self._plot_queue,))
            self._plot_process.start()
            self._plot_background_sent = False
            self._graph_button.text = "Hide Graphs"

        else:

            # Drains the queue before sending shutdown signal.
            if self._plot_queue is not None:

                try:
                    while not self._plot_queue.empty():
                        self._plot_queue.get_nowait()
                except Empty:
                    pass

                self._plot_queue.put(None)

            if self._plot_process is not None:

                self._plot_process.join(timeout=1)
                if self._plot_process.is_alive():
                    self._plot_process.terminate()

            self._plot_process = None
            self._plot_queue = None
            self._graph_button.text = "Show Graphs"

    def _send_plot_data(self) -> None:

        """
        Collects stats and sends to plotting process if open.
        """

        fitness_values: list[float] = [c.fitness for c in self._controllers]
        checkpoints: list[int] = [c.car.current_checkpoint for c in self._controllers]
        laps: list[int] = [c.car.laps_completed for c in self._controllers]
        survival_times: list[float] = [c.time_alive for c in self._controllers]

        # Collects death positions from cars that died this generation.
        death_positions: list[tuple[float, float]] = [
            (c.car.position.x, c.car.position.y)
            for c in self._controllers if not c.is_alive
        ]

        # Always collects history.
        self._plot_history['generations'].append(self.genetic_algorithm.generation - 1)
        self._plot_history['best_fitness'].append(max(fitness_values))
        self._plot_history['avg_fitness'].append(sum(fitness_values) / len(fitness_values))
        self._plot_history['worst_fitness'].append(min(fitness_values))
        self._plot_history['death_positions'].extend(death_positions)

        # Only sends information if the plot window is open.
        if self._plot_queue is None:
            return

        data: dict[str, Any] = {
            # Full history.
            'generations': self._plot_history['generations'].copy(),
            'best_fitness': self._plot_history['best_fitness'].copy(),
            'avg_fitness': self._plot_history['avg_fitness'].copy(),
            'worst_fitness': self._plot_history['worst_fitness'].copy(),
            'death_positions': self._plot_history['death_positions'].copy(),

            # Current generation data.
            'current_gen': self.genetic_algorithm.generation - 1,
            'fitness_distribution': fitness_values,
            'checkpoints': checkpoints,
            'laps': laps,
            'survival_times': survival_times
        }

        # Includes track background on first send after plot window opens.
        if not self._plot_background_sent:

            data['track_bg'] = pygame.surfarray.array3d(self._track.background).transpose(1, 0, 2)  # type: ignore
            self._plot_background_sent = True

        self._plot_queue.put(data)

    def _toggle_mode(self) -> None:

        """
        Toggles between visual and console mode.
        """

        self._visual_mode = not self._visual_mode

        if self._visual_mode:

            self._toggle_button.rect.topleft = (GAME.SCREEN_WIDTH - 220, 20)
            self._toggle_button.text = "Hide Training"
            self._stop_button.rect.topleft = (GAME.SCREEN_WIDTH - 220, 65)
            self._graph_button.rect.topleft = (GAME.SCREEN_WIDTH - 220, 110)

        else:

            self._toggle_button.rect.center = ((GAME.SCREEN_WIDTH // 2), GAME.SCREEN_HEIGHT // 2)
            self._toggle_button.text = "Show Training"
            self._stop_button.rect.center = ((GAME.SCREEN_WIDTH // 2), GAME.SCREEN_HEIGHT // 2 + 45)
            self._graph_button.rect.center = ((GAME.SCREEN_WIDTH // 2), GAME.SCREEN_HEIGHT // 2 + 90)

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
        for controller in self._controllers:

            if not controller.is_alive:
                continue

            if run_ai:
                controller.update_sensors(self._track)
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

        car: Car = controller.car

        # Checks for collisions with the track bounds using the fast method.
        if car.check_track_collision(self._track):
            controller.kill()
            return

        # Checks for checkpoint crossing using fast bounding box pre-rejection.
        checkpoint_order: int = self._track.check_checkpoint(car.position.x, car.position.y)

        if checkpoint_order >= 0:
            controller.handle_checkpoint_hit(checkpoint_order, self._num_checkpoints)

        # Checks for finish line crossing.
        if car.rect.colliderect(self._track.finish_line):
            controller.handle_finish_line(self._num_checkpoints)

    def _print_console_status(self) -> None:

        """
        Prints current generation status to the console.
        """

        alive_count: int = sum(c.is_alive for c in self._controllers)
        time_remaining: float = max(0.0, TRAINING.MAX_GENERATION_TIME - self._generation_timer)

        if not self._controllers:
            return

        fitness_values: list[float] = [c.fitness for c in self._controllers]
        best_fitness: float = max(fitness_values)
        avg_fitness: float = sum(fitness_values) / len(fitness_values)

        best_controller: AIController = max(self._controllers, key=lambda c: c.fitness)
        best_car: Car = best_controller.car

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

        # Background.
        self._screen.fill(COLOURS.BACKGROUND)

        # Title text.
        draw_outlined_text(
            self._screen,
            "Training Mode: Console Output",
            (GAME.SCREEN_WIDTH // 2, GAME.SCREEN_HEIGHT // 2 - 45),
            text_colour=COLOURS.TEXT_SECONDARY
        )

        # Draws the buttons.
        self._toggle_button.draw(self._screen)
        self._stop_button.draw(self._screen)
        self._graph_button.draw(self._screen)

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
        self._track.draw(self._screen)

        # Draws cars with colour coding.
        if self._controllers:

            fitness_values: list[float] = [c.fitness for c in self._controllers]
            best_fitness: float = max(fitness_values)
            worst_fitness: float = min(fitness_values)

            for controller in self._controllers:

                is_best: bool = (controller.fitness == best_fitness)
                is_worst: bool = (controller.fitness == worst_fitness) and not is_best
                controller.draw(self._screen, is_best, is_worst)

        # Draws stats overlay.
        self._draw_visual_stats_overlay()

        # Draws the buttons.
        self._toggle_button.draw(self._screen)
        self._stop_button.draw(self._screen)
        self._graph_button.draw(self._screen)

        pygame.display.flip()

    def _draw_visual_stats_overlay(self) -> None:

        """
        Draws stats overlay in visual mode.

        Notes
        -----
        Displays the generation number, cars alive, time remaining,
        the best fitness, average fitness, and best car's progress.
        """

        alive_count: int = sum(c.is_alive for c in self._controllers)
        time_remaining: float = max(0.0, TRAINING.MAX_GENERATION_TIME - self._generation_timer)

        if self._controllers:

            fitness_values: list[float] = [c.fitness for c in self._controllers]
            best_fitness: float = max(fitness_values)
            avg_fitness: float = sum(fitness_values) / len(fitness_values)

            best_controller: AIController = max(self._controllers, key=lambda c: c.fitness)
            best_car: Car | None = best_controller.car

        else:

            best_fitness = avg_fitness = 0
            best_car = None

        # Semi-transparent background.
        overlay = pygame.Surface((350, 160))
        overlay.set_alpha(200)
        overlay.fill(COLOURS.ITEM_UNSELECTED)
        self._screen.blit(overlay, (10, 10))

        # Stats.
        y = 20
        draw_outlined_text(
            self._screen, f"Generation: {self.genetic_algorithm.generation}",
            (20, y), align="left", font_size=FONTS.SIZE_NORMAL
        )

        y += 25
        draw_outlined_text(
            self._screen, f"Alive: {alive_count}/{TRAINING.POPULATION_SIZE}",
            (20, y), align="left", font_size=FONTS.SIZE_NORMAL
        )

        y += 25
        draw_outlined_text(
            self._screen, f"Time: {time_remaining:.1f}s / {TRAINING.MAX_GENERATION_TIME:.0f}s",
            (20, y), align="left", font_size=FONTS.SIZE_NORMAL
        )

        y += 25
        draw_outlined_text(
            self._screen, f"Best Fitness: {best_fitness:.0f}",
            (20, y), align="left", font_size=FONTS.SIZE_NORMAL
        )

        y += 25
        draw_outlined_text(
            self._screen, f"Avg Fitness: {avg_fitness:.0f}",
            (20, y), align="left", font_size=FONTS.SIZE_NORMAL
        )

        if best_car:

            y += 25
            draw_outlined_text(
                self._screen,
                f"Best CP: {best_car.current_checkpoint}/{self._num_checkpoints} | "
                f"Laps: {best_car.laps_completed}",
                (20, y), align="left", font_size=FONTS.SIZE_NORMAL
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

        all_dead: bool = all(not c.is_alive for c in self._controllers)
        time_up: bool = self._generation_timer >= TRAINING.MAX_GENERATION_TIME

        return all_dead or time_up

    def _next_generation(self) -> None:

        """
        Evolves to the next generation and saves genomes.

        Notes
        -----
        This method calculates fitness statistics, evolves the population,
        saves the best genomes, and creates the next generation.
        """

        genome_to_controller: dict[int, AIController] = {id(c.genome): c for c in self._controllers}

        def fitness_func(g: Genome) -> float:

            c = genome_to_controller.get(id(g))
            return c.fitness if c else 0.0

        # Evolves the population.
        self.genetic_algorithm.next_generation(fitness_func)

        # Calculates statistics.
        fitness_values: list[float] = [c.fitness for c in self._controllers]
        avg_fitness: float = sum(fitness_values) / len(fitness_values)
        max_fitness: float = max(fitness_values)
        min_fitness: float = min(fitness_values)

        self._total_generations += 1

        best_controller: AIController = max(self._controllers, key=lambda c: c.fitness)
        best_car: Car = best_controller.car

        gen_num: int = self.genetic_algorithm.generation - 1

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

        # Periodic autosave.
        if gen_num % self._save_interval == 0 and gen_num > 0:

            GenomeIO.save_best_genomes(
                self.genetic_algorithm,
                num_best=TRAINING.SAVE_AMOUNT,
                directory='./data/genomes'
            )

            print(f"  Autosaved top 3 genomes for generation {gen_num}.")

        print(f"{'=' * 60}")

        # Sends the generation data to the plot.
        self._send_plot_data()

        # Creates a new generation.
        self._create_generation()

    def _print_final_stats(self) -> None:

        """
        Prints final statistics when training stops.
        """

        print(f"Total Generations Completed: {self._total_generations}")
        print("=" * 60)
