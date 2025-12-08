import numpy as np

from numpy.typing import NDArray
from algorithm.config import GENETIC
from algorithm.genome import Genome
from algorithm.neural_network import NeuralNetwork
from game.car import Car


class AIController:

    """
    Controls a car using a neural network.

    Attributes
    ----------
    car : Car
        The car being controlled.
    genome : Genome
        The genome defining the neural network.
    network : NeuralNetwork
        The neural network for decision-making.
    """

    def __init__(self, car: Car, genome: Genome) -> None:

        self.car: Car = car
        self.genome: Genome = genome
        self.network: NeuralNetwork = NeuralNetwork.from_genome(genome)
        self.fitness: float = 0.0

        self.car.is_ai_controlled = True

        self._time_alive: float = 0.0

    def update(self, dt: float) -> None:

        """
        Updates the AI controller.

        Parameters
        ----------
        dt : float
            Time since last update, in seconds.
        """

        if not self.car.is_alive:
            return

        self._time_alive += dt

        inputs: NDArray[float] = np.array(self.car.sensor_distances)
        outputs = self.network.forward(inputs)

        # Retrieves the acceleration and braking probabilities.
        # These will be mutually exclusive.
        accel_prob: float = outputs[0]
        brake_prob: float = outputs[1]

        # Retrieves the probability of turning right or left.
        turn_left_prob: float = outputs[2]
        turn_right_prob: float = outputs[3]

        # Chooses which action(s) to take.
        accelerate: bool = (accel_prob > brake_prob)
        brake: bool = (brake_prob >= accel_prob)
        turn_left = (turn_left_prob > 0.5)
        turn_right = (turn_right_prob > 0.5)
        turn = int(turn_right) - int(turn_left)  # 0 if both or neither are True.

        self.car.set_ai_controls(accelerate, brake, turn)

    def calculate_fitness(self) -> float:

        """
        Calculates the fitness score for this controller.

        Returns
        -------
        float
            The fitness score.

        Notes
        -----
        Fitness is based on checkpoint reached, laps completed, and time alive.
        Rewards are controlled by ``GeneticConfig.REWARD_CHECKPOINT``,
        ``GeneticConfig.REWARD_LAP``, and ``GENETIC.REWARD_TIME``.
        """

        checkpoint_score: float = self.car.current_checkpoint * GENETIC.REWARD_CHECKPOINT
        lap_score: float = self.car.laps_completed * GENETIC.REWARD_LAP
        time_score: float = self._time_alive * GENETIC.REWARD_TIME

        self.fitness = checkpoint_score + lap_score + time_score
        return self.fitness
