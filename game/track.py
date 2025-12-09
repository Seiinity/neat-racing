from __future__ import annotations

import pytmx
import pygame

from game.core.utils import draw_outlined_text, get_tiled_layer
from game.events import Events
from pygame.math import Vector2
from pygame.rect import Rect
from pygame.surface import Surface
from pytmx import TiledImageLayer, TiledObjectGroup, TiledObject, TiledMap
from game.config import TRACK
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate


class Track:

    """
    Represents a racing track loaded from a Tiled map file.

    Attributes
    ----------
    checkpoints : list[Checkpoint]
        List of checkpoints on the track.
    finish_line : Rect
        The finish line rectangle.
    start_positions : list[Vector2]
        A list of starting position for AI cars.
    player_start_position : Vector2
        The starting position for the player car.
    shape : Polygon
        The track's valid racing area.

    Methods
    -------
    draw(self, screen: Surface) -> None
        Draws the track on the screen.
    is_on_track(self, position: Vector2) -> bool
        Checks if a position is within the track boundaries.
    """

    def __init__(self, tmx_path: str):

        self._tmx_data: TiledMap = pytmx.load_pygame(tmx_path)

        self._width: int = self._tmx_data.width * self._tmx_data.tilewidth
        self._height: int = self._tmx_data.height * self._tmx_data.tileheight
        self._background: Surface = self._load_background()
        self._show_checkpoints: bool = False

        self.checkpoints: list[Checkpoint] = self._load_checkpoints()
        self.checkpoints.sort(key=lambda cp: cp.order)

        self.finish_line: Rect = self._load_finish_line()
        self.shape: Polygon = self._load_shape()

        self.start_positions: list[Vector2] = []
        self.player_start_position: Vector2 = Vector2(0, 0)
        self._load_start_positions()

        self._add_listeners()

    def _load_background(self) -> Surface:

        """
        Loads and scales the background image from the Tiled map.

        Returns
        -------
        Surface
            The scaled background surface.

        Notes
        -----
        The background image is loaded from the layer of the Tiled map
        with name ``bg``.
        """

        bg_layer: TiledImageLayer | None = get_tiled_layer(self._tmx_data, 'bg')
        return pygame.transform.scale(bg_layer.image, (self._width, self._height))

    def _load_checkpoints(self) -> list[Checkpoint]:

        """
        Loads all the checkpoint objects from the Tiled map.

        Returns
        -------
        list[Checkpoint]
            A list of checkpoint instances.

        Notes
        -----
        Checkpoints are loaded from the layer of the Tiled map with
        name ``objects`` and must have the class ``checkpoint``.
        """

        object_layer: TiledObjectGroup | None = get_tiled_layer(self._tmx_data, 'objects')
        return [Checkpoint(obj) for obj in object_layer if obj.type == 'checkpoint']

    def _load_finish_line(self) -> Rect:

        """
        Loads the finish line object from the Tiled map.

        Returns
        -------
        Rect
            The finish line's bounding rectangle.

        Notes
        -----
        The finish line is loaded from the layer of the Tiled map
        with name ``objects`` and must have the name ``finish_line``.
        """

        object_layer: TiledObjectGroup | None = get_tiled_layer(self._tmx_data, 'objects')
        finish_line_obj: TiledObject = next(obj for obj in object_layer if obj.name == 'finish_line')
        return Rect(finish_line_obj.x, finish_line_obj.y, finish_line_obj.width, finish_line_obj.height)

    def _load_start_positions(self) -> None:

        """
        Loads the starting positions from the Tiled map.

        Notes
        -----
        Starting positions are loaded from the layer of the Tiled map
        with name ``objects``. The player's starting position must have the
        type ``start_pos_player``. The AI's starting positions must have
        the type ``start_pos``.
        """

        object_layer: TiledObjectGroup | None = get_tiled_layer(self._tmx_data, 'objects')
        player_start_pos_obj = next(obj for obj in object_layer if obj.type == 'start_pos_player')

        self.start_positions = [Vector2(obj.x, obj.y) for obj in object_layer if obj.type == 'start_pos']
        self.player_start_position = Vector2(player_start_pos_obj.x, player_start_pos_obj.y)

    def _load_shape(self) -> Polygon:

        """
        Loads the track's shape from the Tiled map's boundary polygons.

        Returns
        -------
        Polygon
            A polygon representing the valid racing area.

        Notes
        -----
        The track shape is created by subtracting the inner boundary from
        the outer boundary. Both polygons are loaded from the layer of the
        Tiled map with name ``bounds``.
        """

        bounds_layer: TiledObjectGroup | None = get_tiled_layer(self._tmx_data, 'bounds')

        polygons = [obj for obj in bounds_layer if hasattr(obj, 'points')]
        outer_bound = polygons[0]
        inner_bound = polygons[1]

        outer_points = [(p.x, p.y) if hasattr(p, 'x') else p for p in outer_bound.points]
        inner_points = [(p.x, p.y) if hasattr(p, 'x') else p for p in inner_bound.points]

        outer_polygon = Polygon(outer_points)
        inner_polygon = Polygon(inner_points)

        return outer_polygon.difference(inner_polygon)

    def _add_listeners(self) -> None:

        """
        Adds methods as event listeners.
        """

        Events.on_keypress_checkpoints.add_listener(self._toggle_checkpoints)

    def is_on_track(self, position: Vector2) -> bool:

        """
        Checks if a position is within the track boundaries.

        Parameters
        ----------
        position : Vector2
            The position to check, in pixels.

        Returns
        -------
        bool
            ``True`` if the position is within the track boundaries,
            ``False`` otherwise.
        """

        point = Point(position.x, position.y)
        return self.shape.contains(point)

    def draw(self, screen: Surface) -> None:

        """
        Draws the track on the screen.

        Parameters
        ----------
        screen : Surface
            The Pygame surface to draw on.

        Notes
        -----
        Draws the background and checkpoints if visibility is toggled on.
        """

        screen.blit(self._background, (0, 0))

        if not self._show_checkpoints:
            return

        for checkpoint in self.checkpoints:
            checkpoint.draw(screen)

    def _toggle_checkpoints(self) -> None:

        """
        Toggles the visibility of checkpoints on the track.
        """

        self._show_checkpoints = not self._show_checkpoints


class Checkpoint:

    """
    Represents a checkpoint on the racing track.

    Attributes
    ----------
    order : int
        The sequential order of this checkpoint in the track.
    shape : Polygon
        The checkpoint's collision shape as a buffered polygon.

    Methods
    -------
    draw(self, screen: Surface) -> None
        Draws the checkpoint on the screen.
    """

    def __init__(self, obj: TiledObject) -> None:

        self.order: int = obj.properties.get('order')
        self.shape: Polygon = Checkpoint._load_shape(obj)

        self._surface: Surface | None = None
        self._surface_rect: Rect | None = None

        self._load_surface(obj)

    def _load_surface(self, obj: TiledObject) -> None:

        """
        Creates the checkpoint's surface for rendering.

        Parameters
        ----------
        obj : TiledObject
            The Tiled object containing position, dimensions, and rotation data.

        Notes
        -----
        The checkpoint colour is controlled by ``TrackConfig.CHECKPOINT_COLOUR``.
        """

        surface = pygame.Surface((obj.width, obj.height), pygame.SRCALPHA)
        surface.fill(TRACK.CHECKPOINT_COLOUR)

        self._surface = pygame.transform.rotate(surface, -obj.rotation)

        # Pygame's rotate() shifts the pivot point every 90 degrees.
        # This compensates with a y offset for rotations in 90-179 and 270-359.
        normalised = obj.rotation % 360
        needs_offset = (int(normalised - 1) // 90) % 2 == 1
        y_offset = -self._surface.get_height() if needs_offset else 0

        # Calculates the position and dimensions of the surface.
        self._surface_rect = self._surface.get_rect(topleft=(obj.x, obj.y + y_offset))

    @staticmethod
    def _load_shape(obj: TiledObject) -> Polygon:

        """
        Creates the checkpoint's collision shape as a polygon.

        Parameters
        ----------
        obj : TiledObject
            The Tiled object containing position, dimensions, and rotation data.

        Returns
        -------
        Polygon
            A  polygon for collision detection.

        Notes
        -----
        The shape is buffered by 5 pixels to guarantee collisions at fast speeds.
        """

        # Creates an accurate rotated polygon for collision checks.
        rect = Polygon([
            (obj.x, obj.y),
            (obj.x + obj.width, obj.y),
            (obj.x + obj.width, obj.y + obj.height),
            (obj.x, obj.y + obj.height)
        ])

        # Rotates and buffers the shape.
        # Buffering increases the size of the shape, which guarantees collisions at fast speeds.
        return rotate(rect, obj.rotation, origin=(obj.x, obj.y)).buffer(5)

    def draw(self, screen: Surface) -> None:

        """
        Draws the checkpoint on the screen.

        Parameters
        ----------
        screen : Surface
            The Pygame surface to draw on.

        Notes
        -----
        Draws both the checkpoint surface and its order number at the centre.
        """

        screen.blit(self._surface, self._surface_rect)
        draw_outlined_text(screen, str(self.order), self._surface_rect.center)

    def get_centre(self) -> Vector2:

        """
        Returns the centre of the checkpoint.

        Returns
        -------
        Vector2
            A Vector2 representing the centre point of the checkpoint.
        """

        return Vector2(self._surface_rect.center)
