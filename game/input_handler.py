import pygame

from game.config import INPUT
from game.events import Events

class InputHandler:

    """
    Class which handles player inputs.

    Methods
    -------
    update() -> None (static)
        Broadcasts events based on the keys pressed by the player.
    """

    @staticmethod
    def update() -> None:

        keys = pygame.key.get_pressed()

        if keys[INPUT.KEY_ACCELERATE]:
            Events.on_keypress_accelerate.broadcast()

        if keys[INPUT.KEY_BRAKE]:
            Events.on_keypress_brake.broadcast()

        if keys[INPUT.KEY_TURN_LEFT]:
            Events.on_keypress_turn.broadcast(data=-1)

        if keys[INPUT.KEY_TURN_RIGHT]:
            Events.on_keypress_turn.broadcast(data=1)