import numpy as np

from numpy.typing import NDArray
from pygame import Color, Surface
from algorithm.config import GENETIC
from algorithm.genome import Genome
from algorithm.neural_network import NeuralNetwork
from game.car import Car
from game.events import Events


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
    """

    def __init__(self, car: Car, genome: Genome) -> None:

        self.car: Car = car
        self.genome: Genome = genome
        self.fitness: float = 0.0
        self.is_alive: bool = True

        self._network: NeuralNetwork = NeuralNetwork.from_genome(genome)
        self._time_alive: float = 0.0
        self._total_distance: float = 0.0
        self._wrong_checkpoints: int = 0
        self._current_actions: tuple[bool, bool, int] = (False, False, 0)

        self._add_listeners()

    def _add_listeners(self) -> None:

        """
        Adds methods as event listeners.
        """

        Events.on_checkpoint_hit.add_listener(self._handle_checkpoint_hit)
        Events.on_car_die.add_listener(self._kill)

    def _handle_checkpoint_hit(self, data: tuple[Car, int, int]) -> None:

        """
        Penalises hitting checkpoints that are 2+ positions away.

        Parameters
        ----------
        data : tuple[Car, int, int]
            Tuple containing the car instance, checkpoint order, and total checkpoints.
        """

        car, checkpoint, total_checkpoints = data

        if car is not self.car:
            return

        # Only incorrect checkpoints are penalised.
        if checkpoint == self.car.current_checkpoint:
            return

        # Calculates the circular "distance" between checkpoints.
        forward_dist = (checkpoint - self.car.current_checkpoint) % total_checkpoints
        backward_dist = (self.car.current_checkpoint - checkpoint) % total_checkpoints
        min_dist = min(forward_dist, backward_dist)

        # Only penalises if 2+ checkpoints away.
        if min_dist >= 2:
            self._wrong_checkpoints += 1

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

        self._time_alive += dt

        # Tracks total distance (only forward movement counts).
        if self.car.velocity > 0:
            self._total_distance += self.car.velocity * dt

        # Runs the neural network.
        inputs: NDArray[float] = np.array(self.car.sensor_distances)
        outputs = self._network.forward(inputs)

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
        self.car.set_ai_controls(accelerate, brake, turn)

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
        distance_score = self._total_distance * GENETIC.REWARD_DISTANCE

        # Rewards moving forward and penalises moving backward.
        checkpoint_score = (
            self.car.current_checkpoint * GENETIC.REWARD_CHECKPOINT +
            self._wrong_checkpoints * GENETIC.PENALTY_WRONG_CHECKPOINT
        )

        # Rewards completing laps.
        lap_score = self.car.laps_completed * GENETIC.REWARD_LAP

        # Rewards staying away from walls.
        avg_sensor_distance = np.mean(self.car.sensor_distances)
        safety_score = avg_sensor_distance * GENETIC.REWARD_SAFETY

        # Rewards maintaining speed.
        velocity_score = max(0.0, self.car.velocity) * GENETIC.REWARD_VELOCITY

        # Time alive is penalised to prevent cars from driving aimlessly.
        survival_score = self._time_alive * GENETIC.PENALTY_TIME

        self.fitness = (
            distance_score +
            checkpoint_score +
            lap_score +
            safety_score +
            velocity_score +
            survival_score
        )

    def _kill(self, data: Car) -> None:

        """
        Kills this AI controller.

        Parameters
        ----------
        data : Car
            The car passed by the car death event.

        Notes
        -------
        A final fitness score is calculated once a controller
        is killed.
        """

        if data is not self.car:
            return

        self._calculate_fitness()
        self.is_alive = False
        self.car.velocity = 0

    def draw(self, screen: Surface, is_best: bool = False, is_worst: bool = False) -> None:

        """
        Draws the AI-controlled car.

        Notes
        -----
        The cars with the absolute lowest fitness scores are drawn
        in red. The cars with the absolute highest fitness scores are
        drawn in green. All other cars are drawn in white.
        """

        colour: Color = Color(255, 255, 255)

        if is_best:
            colour = Color(0, 255, 0)
        elif is_worst:
            colour = Color(255, 0, 0)

        self.car.draw(screen, colour)

    def dispose(self) -> None:

        """
        Liberates the AI controller for garbage collection.
        """

        Events.on_checkpoint_hit.remove_listener(self._handle_checkpoint_hit)
        Events.on_car_die.remove_listener(self._kill)
        self.car.dispose()
