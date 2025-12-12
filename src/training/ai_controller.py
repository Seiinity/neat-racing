import numpy as np

from numpy.typing import NDArray
from pygame import Color, Surface, Vector2
from pygame.draw import line
from config import CONTROLLER, FITNESS
from src.algorithm import Genome, NeuralNetwork
from src.core import Car, Track, Events


class AIController:

    """
    Controls a car using a neural network.

    Attributes
    ----------
    car : Car
        The car being controlled.
    genome : Genome
        The genome defining the neural network.
    fitness : float
        The fitness of the genome.
    is_alive : bool
        Whether the controller is currently alive or not.
    time_alive : float
        Time spent alive, in seconds.

    Methods
    -------
    handle_checkpoint_hit(checkpoint_order: int, total_checkpoints: int) -> None
        Handles checkpoint hits.
    handle_finish_line(total_checkpoints: int) -> None
        Handles finish line crossing.
    kill() -> None
        Kills the AI controller.
    make_decision(dt: float) -> None
        Makes a decision regarding what actions to take.
    fixed_update() -> None
        Updates the AI controller.
    draw(screen: Surface, is_best: bool = False, is_worst: bool = False) -> None
        Draws the AI-controlled car.
    dispose() -> None
        Liberates the AI controller for garbage collection.
    """

    def __init__(self, car: Car, genome: Genome) -> None:

        self.car: Car = car
        self.genome: Genome = genome
        self.fitness: float = 0.0
        self.is_alive: bool = True
        self.time_alive: float = 0.0

        self._network: NeuralNetwork = NeuralNetwork.from_genome(genome)
        self._total_distance: float = 0.0
        self._wrong_checkpoints: int = 0
        self._current_actions: tuple[bool, bool, int] = (False, False, 0)

        # Sensor data.
        self._sensor_distances: list[float] = [0.0] * len(CONTROLLER.SENSORS)
        self._show_sensors: bool = False
        self._sensor_angles_rad: list[float] = [np.radians(angle) for angle in CONTROLLER.SENSORS]

        Events.on_keypress_sensors.add_listener(self._toggle_sensors)

    def update_sensors(self, track: Track) -> None:

        """
        Updates all sensor distances using the track's collision mask.

        Parameters
        ----------
        track : Track
            The track to raycast against.
        """

        for i, angle_offset in enumerate(self._sensor_angles_rad):

            # Calculates the sensor's direction.
            sensor_angle: float = self.car.angle + angle_offset
            direction: Vector2 = Vector2(np.cos(sensor_angle), np.sin(sensor_angle))

            # Casts a ray.
            distance: float = track.raycast(self.car.position, direction, CONTROLLER.SENSOR_RANGE)
            self._sensor_distances[i] = distance / CONTROLLER.SENSOR_RANGE

    def handle_checkpoint_hit(self, checkpoint_order: int, total_checkpoints: int) -> None:

        """
        Handles checkpoint hits.

        Parameters
        ----------
        checkpoint_order : int
            The order of the checkpoint that was hit.
        total_checkpoints : int
            The total number of checkpoints on the track.

        Notes
        -----
        Penalises hitting checkpoints that are 2+ positions away.
        Also forwards the event to the car for progression tracking.
        """

        # Forwards to car for checkpoint progression.
        self.car.handle_checkpoint_hit(checkpoint_order)

        # Only incorrect checkpoints are penalised.
        if checkpoint_order == self.car.current_checkpoint - 1:
            return

        # Calculates the circular "distance" between checkpoints.
        # Note: current_checkpoint was already incremented if correct, so we check against previous.
        expected: int = self.car.current_checkpoint - 1 if self.car.current_checkpoint > 0 else 0
        forward_dist: int = (checkpoint_order - expected) % total_checkpoints
        backward_dist: int = (expected - checkpoint_order) % total_checkpoints
        min_dist: int = min(forward_dist, backward_dist)

        # Only penalises if 2+ checkpoints away.
        if min_dist >= 2:
            self._wrong_checkpoints += 1

    def handle_finish_line(self, total_checkpoints: int) -> None:

        """
        Handles finish line crossing.

        Parameters
        ----------
        total_checkpoints : int
            The total number of checkpoints on the track.
        """

        self.car.handle_finish_line(total_checkpoints)

    def kill(self) -> None:

        """
        Kills the AI controller.

        Notes
        -------
        A final fitness score is calculated once a controller
        is killed.
        """

        if not self.is_alive:
            return

        self._calculate_fitness()
        self.is_alive = False
        self.car.velocity = 0

    def make_decision(self, dt: float) -> None:

        """
        Makes a decision regarding what actions to take.

        Parameters
        ----------
        dt : float
            Time since the last AI decisions, in seconds.
        """

        if not self.is_alive:
            return

        self.time_alive += dt

        # Tracks total distance (only forward movement counts).
        if self.car.velocity > 0:
            self._total_distance += self.car.velocity * dt

        # Runs the neural network.
        inputs: NDArray[float] = np.array(self._sensor_distances)
        outputs: NDArray[float] = self._network.forward(inputs)

        # Retrieves the acceleration and braking probabilities.
        accel_prob: float = outputs[0]
        brake_prob: float = outputs[1]

        # Retrieves the probability of turning right or left.
        turn_left_prob: float = outputs[2]
        turn_right_prob: float = outputs[3]

        # Chooses which action(s) to take.
        accelerate: bool = (accel_prob > brake_prob)
        brake: bool = (brake_prob >= accel_prob)
        turn = int(turn_right_prob > 0.5) - int(turn_left_prob > 0.5)

        # Saves the actions to take.
        self._current_actions = (accelerate, brake, turn)

    def fixed_update(self) -> None:

        """
        Updates the AI controller.
        """

        if not self.is_alive:
            return

        self._calculate_fitness()

        # Applies the AI's decisions.
        accelerate, brake, turn = self._current_actions

        if accelerate:
            Events.on_keypress_accelerate.broadcast(data=self.car)

        if brake:
            Events.on_keypress_brake.broadcast(data=self.car)

        if turn != 0:
            Events.on_keypress_turn.broadcast(data=(self.car, turn))

    def _calculate_fitness(self) -> None:

        """
        Calculates the fitness score for this controller.

        Notes
        -----
        Fitness is calculated based on distance traveled, checkpoints
        crossed, laps completed, distance to walls, velocity, and time
        alive.
        """

        # Rewards distance travelled.
        distance_score: float = self._total_distance * FITNESS.REWARD_DISTANCE

        # Rewards moving forward and penalises moving backward.
        checkpoint_score: float = (
            self.car.current_checkpoint * FITNESS.REWARD_CHECKPOINT +
            self._wrong_checkpoints * FITNESS.PENALTY_WRONG_CHECKPOINT
        )

        # Rewards completing laps.
        lap_score: float = self.car.laps_completed * FITNESS.REWARD_LAP

        # Rewards staying away from walls.
        avg_sensor_distance: float = float(np.mean(self._sensor_distances))
        safety_score: float = avg_sensor_distance * FITNESS.REWARD_SAFETY

        # Rewards maintaining speed.
        velocity_score: float = max(0.0, self.car.velocity) * FITNESS.REWARD_VELOCITY

        # Time alive is penalised to prevent cars from driving aimlessly.
        time_penalty: float = self.time_alive * FITNESS.PENALTY_TIME

        self.fitness = (
            distance_score +
            checkpoint_score +
            lap_score +
            safety_score +
            velocity_score +
            time_penalty
        )

    def draw(self, screen: Surface, is_best: bool = False, is_worst: bool = False) -> None:

        """
        Draws the AI-controlled car.

        Notes
        -----
        The cars with the absolute lowest fitness scores are drawn
        in red. The cars with the absolute highest fitness scores are
        drawn in green. All other cars are drawn in their default colour.
        """

        colour: Color | None = None

        if is_best:
            colour = Color(0, 255, 0)
        elif is_worst:
            colour = Color(255, 0, 0)

        self.car.draw(screen, colour)

        if self._show_sensors:
            self._draw_sensors(screen)

    def _draw_sensors(self, screen: Surface) -> None:

        """
        Draws the car's sensors for debug purposes.

        Parameters
        ----------
        screen : Surface
            The screen to draw on.
        """

        for i, angle_offset in enumerate(self._sensor_angles_rad):

            sensor_angle: float = self.car.angle + angle_offset
            direction: Vector2 = Vector2(np.cos(sensor_angle), np.sin(sensor_angle))
            distance: float = self._sensor_distances[i] * CONTROLLER.SENSOR_RANGE
            end_point: Vector2 = self.car.position + direction * distance

            line(
                screen,
                Color('#9ee88b'),
                (self.car.position.x, self.car.position.y),
                (end_point.x, end_point.y),
                1
            )

    def _toggle_sensors(self) -> None:

        """
        Toggles the visualisation of sensors.
        """

        self._show_sensors = not self._show_sensors

    def dispose(self) -> None:

        """
        Liberates the AI controller for garbage collection.
        """

        Events.on_keypress_sensors.remove_listener(self._toggle_sensors)

        self.car.dispose()
