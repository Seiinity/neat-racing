import pytmx
import pygame

from pygame.math import Vector2
from pygame.rect import Rect
from pygame.surface import Surface
from pytmx import TiledImageLayer, TiledObjectGroup, TiledObject, TiledMap

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
        print(Vector2(start_pos_obj.x, start_pos_obj.y))
        return Vector2(start_pos_obj.x, start_pos_obj.y)

    def draw(self, screen: Surface) -> None:

        screen.blit(self.background, (0, 0))
        for checkpoint in self.checkpoints:
            checkpoint.draw(screen)


class Checkpoint:

    def __init__(self, obj: TiledObject):

        self.order: int = obj.properties.get('order')
        self.rect: Rect = Rect(obj.x, obj.y, obj.width, obj.height)
        self.rotation: float = obj.rotation
        self.x: float = obj.x
        self.y: float = obj.y
        self.width: float = obj.width
        self.height: float = obj.height

        self.surface_orig = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.surface_orig.fill((255, 255, 0, 128))  # Yellow, semi-transparent

    def draw(self, screen: Surface, color=(255, 255, 0)) -> None:

        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surface.fill((*color, 128))

        rotated_surface = pygame.transform.rotate(surface, -self.rotation)

        screen.blit(rotated_surface, (self.x, self.y))