import pygame

from pygame.surface import Surface
from game.config import TRACK

class Track:

    def __init__(self, image_path: str):

        original: Surface = pygame.image.load(image_path).convert()
        self.image: Surface = pygame.transform.scale(original, (TRACK.WIDTH, TRACK.HEIGHT))

    def draw(self, screen: Surface) -> None:

        screen.blit(self.image, (0, 0))