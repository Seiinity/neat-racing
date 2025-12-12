from __future__ import annotations

import pygame
import numpy as np

from pygame import Color, Surface, Vector2, Rect
from shapely import Point
from shapely.geometry.multilinestring import MultiLineString

from config import CAR, COLOURS
from .events import Events
from .track import Track


class Car:

    """
    Represents a car in the game world.

    Attributes
    ----------
    rect : Rect
        The car's bounding rectangle.
    position : Vector2
        The current position of the car, in pixels.
    velocity : float
        The current velocity of the car, in pixels/s.
    current_checkpoint : int
        The order of the checkpoint the car must hit next.
    laps_completed : int
        The number of laps completed by the car.
    sensor_distances : list[float]
        A list containing the distances of all sensors to walls.

    Methods
    -------
    fixed_update(dt: float) -> None
        Updates the car using physics operations.
    update_sensors(track: Track) -> None
        Updates all sensor distances using the track's collision mask.
    check_track_collision(track: Track) -> bool
        Checks whether any point of the car is off the track.
    handle_checkpoint_hit(checkpoint_order: int) -> None
         Handles checkpoint collision.
    handle_finish_line(total_checkpoints: int) -> None
        Handles finish line crossing.
    draw(screen: Surface) -> None
        Draws the car on the screen.
    get_transformed_points(part: str) -> list[Vector2]
        Gets the points that make up the car's shape.
    dispose() -> None
        Liberates the car for garbage collection.
    """

    def __init__(self, start_pos: Vector2, colour: Color = COLOURS.CAR_DEFAULT) -> None:

        self.rect: Rect | None = None
        self.position: Vector2 = start_pos.copy()
        self.velocity: float = 0.0
        self._angle: float = 0.0
        self._colour: Color = colour

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

        # Sensor data.
        self.sensor_distances: list[float] = [0.0] * len(CAR.SENSORS)
        self._show_sensors: bool = False

        # Pre-computes sensor angle offsets in radians for performance.
        self._sensor_angles_rad: list[float] = [np.radians(angle) for angle in CAR.SENSORS]

        self._update_rect()
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

        Acceleration is controlled by ``CarConfig.ACCELERATION``.
        Brake strength is controlled by ``CarConfig.BRAKE_STRENGTH``.
        Turn speed is controlled by ``CarConfig.TURN_SPEED``.
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
        self._angle += self._turn_direction * np.radians(CAR.TURN_SPEED) * dt

        # Applies friction to the car's velocity.
        # Friction is already a multiplicative decay, so it needs to be normalised.
        self.velocity *= CAR.FRICTION ** (dt * 60)

        # Changes the car's position based on direction and velocity.
        self._direction = Vector2(np.cos(self._angle), np.sin(self._angle))
        self.position += self._direction * self.velocity * dt

        # Updates the car rect.
        self._update_rect()

        # Resets inputs.
        self._reset_input()

    def _update_rect(self) -> None:

        """
        Updates the bounding rect based on the actual triangle points.
        """

        points: list[Vector2] = self.get_transformed_points('triangle')
        xs: list[float] = [p.x for p in points]
        ys: list[float] = [p.y for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        self.rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

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
            sensor_angle: float = self._angle + angle_offset
            direction: Vector2 = Vector2(np.cos(sensor_angle), np.sin(sensor_angle))

            # Casts a ray.
            distance: float = track.raycast(self.position, direction, CAR.SENSOR_RANGE)
            self.sensor_distances[i] = distance / CAR.SENSOR_RANGE

    def check_track_collision(self, track: Track) -> bool:

        """
        Checks whether any point of the car is off the track.

        Parameters
        ----------
        track : Track
            The track to check against.

        Returns
        -------
        bool
            ``True`` if any point is off the track (collision),
            ``False`` otherwise.
        """

        for point in self.get_transformed_points('triangle'):
            if not track.is_on_track(int(point.x), int(point.y)):
                return True

        return False

    def handle_checkpoint_hit(self, checkpoint_order: int) -> None:

        """
        Handles checkpoint collision.

        Parameters
        ----------
        checkpoint_order : int
            The order of the checkpoint that was hit.
        """

        # Only counts the needed checkpoint.
        if checkpoint_order != self.current_checkpoint:
            return

        # Increases the current checkpoint.
        self.current_checkpoint += 1

    def handle_finish_line(self, total_checkpoints: int) -> None:

        """
        Handles finish line crossing.

        Parameters
        ----------
        total_checkpoints : int
            The total number of checkpoints on the track.
        """

        # Only counts if all checkpoints have been hit.
        if total_checkpoints != self.current_checkpoint:
            return

        # Increases the number of laps completed and resets checkpoints.
        self.laps_completed += 1
        self.current_checkpoint = 0

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
        boundary: MultiLineString = track.shape.boundary
        nearest: Point = boundary.interpolate(
            boundary.project(Point(self.position.x, self.position.y))
        )

        # Calculates the collision normal.
        normal: Vector2 = Vector2(self.position.x - nearest.x, self.position.y - nearest.y).normalize()

        # Pushes the car away from the wall.
        self.position += normal * CAR.SIZE * 0.1

        # Calculates the velocity when sliding along a wall.
        v = self._direction * self.velocity
        normal_component: Vector2 = normal * v.dot(normal)
        v_sliding: Vector2 = v - normal_component

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

        return all(track.is_on_track(int(p.x), int(p.y)) for p in self.get_transformed_points('triangle'))

    def draw(self, screen: Surface, colour: Color | None = None) -> None:

        """
        Draws the car on the screen.

        Parameters
        ----------
        screen : Surface
            The Pygame surface to draw on.
        colour : Color
            The colour to draw the car in.
        """

        if colour is None:
            colour = self._colour

        # Draws the triangle and line.
        # The line's points must be unpacked as the line function accepts separate arguments.
        pygame.draw.polygon(screen, colour, self.get_transformed_points('triangle'), width=2)
        pygame.draw.line(screen, colour, *self.get_transformed_points('line'), width=2)

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

            sensor_angle: float = self._angle + angle_offset
            direction: Vector2 = Vector2(np.cos(sensor_angle), np.sin(sensor_angle))
            distance: float = self.sensor_distances[i] * CAR.SENSOR_RANGE
            end_point: Vector2 = self.position + direction * distance

            pygame.draw.line(
                screen,
                Color('#9ee88b'),
                (self.position.x, self.position.y),
                (end_point.x, end_point.y),
                1
            )

    def get_transformed_points(self, part: str) -> list[Vector2]:

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
            self.position + Vector2(x * CAR.SIZE, y * CAR.SIZE).rotate_rad(self._angle)
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

        Events.on_keypress_sensors.add_listener(self._toggle_sensors)

    def _toggle_sensors(self) -> None:

        """
        Toggles the visualisation of sensors.
        """

        self._show_sensors = not self._show_sensors

    def _accelerate(self, data: Car) -> None:

        """
        Sets the acceleration flag to True.
        """

        if data is not self:
            return

        self._is_accelerating = True

    def _brake(self, data: Car) -> None:

        """
        Sets the breaking flag to True.
        """

        if data is not self:
            return

        self._is_braking = True

    def _turn(self, data: tuple[Car, int]) -> None:

        """
        Sets the turning direction.

        Parameters
        ----------
        data : int
            Its sign controls the direction of the turning.
        """

        car, direction = data

        if car is not self:
            return

        # Sign makes sure that the parameter only controls the direction, not the speed.
        self._turn_direction = np.sign(direction)

    def _reset_input(self) -> None:

        """
        Resets inputs for next frame.
        """

        self._is_accelerating = False
        self._is_braking = False
        self._turn_direction = 0

    def dispose(self) -> None:

        """
        Liberates the car for garbage collection.
        """

        Events.on_keypress_accelerate.remove_listener(self._accelerate)
        Events.on_keypress_brake.remove_listener(self._brake)
        Events.on_keypress_turn.remove_listener(self._turn)

        Events.on_car_collided.remove_listener(self._handle_collision)

        Events.on_keypress_sensors.remove_listener(self._toggle_sensors)
