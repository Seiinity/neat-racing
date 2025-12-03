import pytmx
import pygame
import numpy as np

from pygame.math import Vector2
from pygame.rect import Rect
from pygame.surface import Surface
from pytmx import TiledImageLayer, TiledObjectGroup, TiledObject, TiledMap
from game.config import TRACK

class Track:

    def __init__(self, tmx_path: str):

        self.tmx_data: TiledMap = pytmx.load_pygame(tmx_path)

        self.width: int = self.tmx_data.width * self.tmx_data.tilewidth
        self.height: int = self.tmx_data.height * self.tmx_data.tileheight

        self.background: Surface = self._load_background()
        self.checkpoints: list[Checkpoint] = self._load_checkpoints()
        self.finish_line: Rect = self._load_finish_line()
        self.start_pos: Vector2 = self._load_start_pos()

    def _load_background(self) -> Surface:

        bg_layer: TiledImageLayer = self.tmx_data.layers[0]
        return pygame.transform.scale(bg_layer.image, (self.width, self.height))

    def _load_checkpoints(self):

        object_layer: TiledObjectGroup = self.tmx_data.layers[1]
        return [Checkpoint(obj) for obj in object_layer if obj.type == 'checkpoint']

    def _load_finish_line(self):

        object_layer: TiledObjectGroup = self.tmx_data.layers[1]
        finish_line_obj: TiledObject = next(obj for obj in object_layer if obj.name == 'finish_line')
        return Rect(finish_line_obj.x, finish_line_obj.y, finish_line_obj.width, finish_line_obj.height)

    def _load_start_pos(self) -> Vector2:

        object_layer: TiledObjectGroup = self.tmx_data.layers[1]
        start_pos_obj: TiledObject = next(obj for obj in object_layer if obj.name == 'start_pos')
        return Vector2(start_pos_obj.x, start_pos_obj.y)

    def draw(self, screen: Surface) -> None:

        screen.blit(self.background, (0, 0))
        for checkpoint in self.checkpoints:
            checkpoint.draw(screen)


class Checkpoint:

    def __init__(self, obj: TiledObject):

        self.order: int = obj.properties.get('order')
        self.x: float = obj.x
        self.y: float = obj.y

        self.surface: Surface | None = None
        self.surface_rect: Rect | None = None
        self.original_surface: Surface | None = None
        self.rotation = 0

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