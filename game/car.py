from __future__ import annotations

import pygame
import numpy as np

from game.config import CAR
from game.events import Events
from game.track import Track
from pygame.surface import Surface
from pygame.math import Vector2
from shapely.geometry import Point


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
    rect : Rect
        The car's Rect, used for checkpoint tracking.

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
        self._update_rect()

        # Progression tracking.
        self.current_checkpoint: int = 0
        self.laps_completed: int = 0

        # Input state flags.
        self._is_accelerating: bool = False
        self._is_braking: bool = False
        self._turn_direction: int = 0

        # Stores previous state for collision handling.
        self._previous_position: Vector2 = start_pos.copy()
        self._previous_velocity: float = 0.0
        self._direction: Vector2 = Vector2(1, 0)

        self._add_listeners()

    def _update_rect(self) -> None:

        """
        Updates the bounding rect based on the actual triangle points.
        """

        points = self._get_transformed_points('triangle')
        xs = [p.x for p in points]
        ys = [p.y for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        self.rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

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

        # Stores the previous position and velocity for collision handling.
        self._previous_position = self.position.copy()
        self._previous_velocity = self.velocity

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
        self._direction = Vector2(np.cos(self.angle), np.sin(self.angle))
        self.position += self._direction * self.velocity * dt
        Events.on_car_moved.broadcast(data=(self, self._get_transformed_points('triangle')))

        # Updates the car Rect.
        self._update_rect()

        # Resets inputs.
        self._reset_input()

    def _handle_collision(self, data: tuple[Car, Track]) -> None:

        """
        Handles collision with track boundaries.

        Parameters
        ----------
        data : tuple[Car, Track]
            Tuple containing the car instance and track that collided.

        Notes
        -----
        Reverts the car to its previous position, calculates the wall normal
        from the track boundary, and pushes it away from the wall.
        """

        car, track = data

        # Only handles collision for this car instance.
        if car is not self:
            return

        # Reverts to the previous safe position.
        self.position = self._previous_position.copy()

        # If no points are off track after reverting, all is good!
        if self._all_points_on_track(track):
            return

        # Finds the track point which is nearest to the car's centre.
        boundary = track.shape.boundary
        nearest = boundary.interpolate(
            boundary.project(Point(self.position.x, self.position.y))
        )

        # Calculates the collision normal.
        normal = Vector2(self.position.x - nearest.x, self.position.y - nearest.y).normalize()

        # Pushes the car away from the wall.
        self.position += normal * CAR.SIZE * 0.1

        # Calculates the velocity when sliding along a wall.
        v = self._direction * self.velocity
        normal_component = normal * v.dot(normal)
        v_sliding = v - normal_component

        # Applies sliding friction.
        v_sliding *= CAR.SLIDING_FRICTION

        # Recomputes the velocity and direction.
        self.velocity = v_sliding.length()
        self._direction = v_sliding.normalize()

    def _all_points_on_track(self, track: Track) -> bool:

        """
        Checks whether all points of the car's triangle shape are inside
        a racing track.

        Parameters
        ----------
        track : Track
            The racing track to check the points on.

        Returns
        -------
        bool
            ``True`` if all points of the car's triangle shape are inside
            the track, ``False`` otherwise.

        """
        return all(track.is_on_track(p) for p in self._get_transformed_points('triangle'))

    def _handle_checkpoint_hit(self, data: tuple[Car, int]) -> None:

        """
        Increases the current checkpoint once the car hits the checkpoint
        it needs to hit next.

        Parameters
        ----------
        data : tuple[Car, int]
            Tuple containing the car instance and order of the hit checkpoint.
        """

        car, checkpoint = data

        # Only handles collision for this car instance.
        if car is not self:
            return

        # Only counts the needed checkpoint.
        if checkpoint is not self.current_checkpoint:
            return

        # Increases the current checkpoint.
        self.current_checkpoint += 1

    def _handle_finish_line_crossed(self, data: tuple[Car, int]) -> None:

        """
        Completes a lap once the car crosses the finish line, but only
        if all checkpoint have been hit beforehand.

        Parameters
        ----------
        data : tuple[Car, int]
            Tuple containing the car instance and the number of checkpoints
            in the racing track.
        """

        car, num_checkpoints = data

        # Only handles collision for this car instance.
        if car is not self:
            return

        # Only counts if all checkpoints have been hit.
        if num_checkpoints is not self.current_checkpoint:
            return

        # Increases the number of laps completed and resets checkpoints.
        self.laps_completed += 1
        self.current_checkpoint = 0

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

        Events.on_car_collided.add_listener(self._handle_collision)
        Events.on_checkpoint_hit.add_listener(self._handle_checkpoint_hit)
        Events.on_finish_line_crossed.add_listener(self._handle_finish_line_crossed)

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
