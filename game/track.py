from __future__ import annotations

import pytmx
import pygame

from game.events import Events
from pygame.math import Vector2
from pygame.rect import Rect
from pygame.surface import Surface
from pytmx import TiledImageLayer, TiledObjectGroup, TiledObject, TiledMap
from game.config import TRACK
from shapely.geometry import Polygon, Point

class Track:

    def __init__(self, tmx_path: str):

        self.tmx_data: TiledMap = pytmx.load_pygame(tmx_path)

        self._width: int = self.tmx_data.width * self.tmx_data.tilewidth
        self._height: int = self.tmx_data.height * self.tmx_data.tileheight
        self._show_checkpoints: bool = False

        self.background: Surface = self._load_background()
        self.checkpoints: list[Checkpoint] = self._load_checkpoints()
        self.finish_line: Rect = self._load_finish_line()
        self.start_pos: Vector2 = self._load_start_pos()
        self.shape: Polygon = self._load_shape()

        self._add_listeners()

    def _load_background(self) -> Surface:

        bg_layer: TiledImageLayer = self.tmx_data.layers[0]
        return pygame.transform.scale(bg_layer.image, (self._width, self._height))

    def _load_checkpoints(self) -> list[Checkpoint]:

        object_layer: TiledObjectGroup = self.tmx_data.layers[1]
        return [Checkpoint(obj) for obj in object_layer if obj.type == 'checkpoint']

    def _load_finish_line(self) -> Rect:

        object_layer: TiledObjectGroup = self.tmx_data.layers[1]
        finish_line_obj: TiledObject = next(obj for obj in object_layer if obj.name == 'finish_line')
        return Rect(finish_line_obj.x, finish_line_obj.y, finish_line_obj.width, finish_line_obj.height)

    def _load_start_pos(self) -> Vector2:

        object_layer: TiledObjectGroup = self.tmx_data.layers[1]
        start_pos_obj: TiledObject = next(obj for obj in object_layer if obj.name == 'start_pos')
        return Vector2(start_pos_obj.x, start_pos_obj.y)

    def _load_shape(self) -> Polygon:

        bounds_layer: TiledObjectGroup = self.tmx_data.layers[2]

        polygons = [obj for obj in bounds_layer if hasattr(obj, 'points')]
        outer_bound = polygons[0]
        inner_bound = polygons[1]

        outer_points = [(p.x, p.y) if hasattr(p, 'x') else p for p in outer_bound.points]
        inner_points = [(p.x, p.y) if hasattr(p, 'x') else p for p in inner_bound.points]

        outer_polygon = Polygon(outer_points)
        inner_polygon = Polygon(inner_points)

        return outer_polygon.difference(inner_polygon)

    def _add_listeners(self) -> None:

        Events.on_keypress_checkpoints.add_listener(self._toggle_checkpoints)

    def draw(self, screen: Surface) -> None:

        screen.blit(self.background, (0, 0))

        if not self._show_checkpoints:
            return

        for checkpoint in self.checkpoints:
            checkpoint.draw(screen)

    def _toggle_checkpoints(self) -> None:

        self._show_checkpoints = not self._show_checkpoints

    def is_on_track(self, position: Vector2) -> bool:

        point = Point(position.x, position.y)
        return self.shape.contains(point)

class Checkpoint:

    def __init__(self, obj: TiledObject):

        self.order: int = obj.properties.get('order')
        self._load_surface(obj)

    def _load_surface(self, obj: TiledObject) -> None:

        surface = pygame.Surface((obj.width, obj.height), pygame.SRCALPHA)
        surface.fill(TRACK.CHECKPOINT_COLOUR)

        self.surface = pygame.transform.rotate(surface, -obj.rotation)

        # Pygame's rotate() shifts the pivot point every 90 degrees.
        # This compensates with a y offset for rotations in 90-179 and 270-359.
        normalised = obj.rotation % 360
        needs_offset = (int(normalised - 1) // 90) % 2 == 1
        y_offset = -self.surface.get_height() if needs_offset else 0

        # Calculates the position and dimensions of the surface.
        self.surface_rect = self.surface.get_rect(topleft=(obj.x, obj.y + y_offset))

    def draw(self, screen: Surface) -> None:

        screen.blit(self.surface, self.surface_rect)