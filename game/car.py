import pygame
import numpy as np
from pygame.surface import Surface
from pygame.math import Vector2
from game.config import CAR
from game.events import Events

class Car:

    """
    Represents a car in the game world.

    Attributes
    ----------
    position : Vector2
        The current position of the car, in pixels.
    velocity : float
        The current velocity of the car, in pixels/s.
    angle : float
        The current angle of the car, in degrees.

    Methods
    -------
    fixed_update(self, dt: float) -> None
        Updates the car using physics operations.
    draw(self, screen: Surface) -> None
        Draws the car on the screen.
    """

    def __init__(self, start_pos: Vector2) -> None:

        self.position: Vector2 = start_pos
        self.velocity: float = 0.0
        self.angle: float = 0.0

        # Input state flags.
        self._is_accelerating: bool = False
        self._is_braking: bool = False
        self._turn_direction: int = 0

        self._add_listeners()

    def fixed_update(self, dt: float) -> None:

        """
        Updates the car.

        Parameters
        ----------
        dt : float
            Fixed timestep duration, in seconds.

        Notes
        -----
        This should be called inside the ``fixed_update()`` method
        of ``GameLoop`` as it contains physics operations.

        Acceleration is controlled by ``CarConfig.ACCELERATION``
        Brake strength is controlled by ``CarConfig.BRAKE_STRENGTH``
        Turn speed is controlled by ``CarConfig.TURN_SPEED``
        """

        # Accelerates.
        if self._is_accelerating:
            self.velocity += CAR.ACCELERATION * dt

        # Brakes.
        if self._is_braking:
            self.velocity -= CAR.BRAKE_STRENGTH * dt

        # Turns.
        self.angle += self._turn_direction * CAR.TURN_SPEED * dt

        # Applies friction to the car's velocity.
        # Friction is already a multiplicative decay, so it needs to be normalised.
        self.velocity *= CAR.FRICTION ** (dt * 60)

        # Changes the car's position based on direction and velocity.
        direction: Vector2 = Vector2(np.cos(self.angle), np.sin(self.angle))
        self.position += direction * self.velocity * dt

        # Resets inputs.
        self._reset_input()

    def draw(self, screen: Surface) -> None:

        """
        Draws the car on the screen.

        Parameters
        ----------
        screen : Surface
            The Pygame surface to draw on.
        """

        # Draws the triangle and line.
        # The line's points must be unpacked as the line function accepts separate arguments.
        pygame.draw.polygon(screen, (255, 255, 255), self._get_transformed_points('triangle'), width=2)
        pygame.draw.line(screen, (255, 255, 255), *self._get_transformed_points('line'), width=2)

    def _get_transformed_points(self, part: str) -> list[Vector2]:

        """
        Gets the points that make up the car's shape.
        The points are scaled according to the car's size and rotated according
        to the car's angle.

        Parameters
        ----------
        part : str
            Which part of the car's shape to get the points for.

        Returns
        -------
        list[Vector2]
            A list of points that make up the car's shape.

        Notes
        -----
        The car's size is controlled by ``CarConfig.SIZE``.
        Its shape is controlled by ``CarConfig.SHAPE``.
        """

        return [
            self.position + Vector2(x * CAR.SIZE, y * CAR.SIZE).rotate_rad(self.angle)
            for x, y in CAR.SHAPE[part]
        ]

    def _add_listeners(self) -> None:

        """
        Adds methods as event listeners.
        """

        Events.on_keypress_accelerate.add_listener(self._accelerate)
        Events.on_keypress_brake.add_listener(self._brake)
        Events.on_keypress_turn.add_listener(self._turn)

    def _accelerate(self) -> None:

        """
        Sets the acceleration flag to True.
        """

        self._is_accelerating = True

    def _brake(self) -> None:

        """
        Sets the breaking flag to True.
        """

        self._is_braking = True

    def _turn(self, data) -> None:

        """
        Sets the turning direction.

        Parameters
        ----------
        data : int
            Its sign controls the direction of the turning.
        """

        # Sign makes sure that the parameter only controls the direction, not the speed.
        self._turn_direction = np.sign(data)

    def _reset_input(self) -> None:

        """
        Resets inputs for next frame.
        """

        self._is_accelerating = False
        self._is_braking = False
        self._turn_direction = 0