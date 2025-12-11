import pygame

from config import INPUT
from src.core import Events


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

        InputHandler._update_held_keys()
        InputHandler._update_pressed_keys()

    @staticmethod
    def _update_held_keys():

        keys = pygame.key.get_pressed()

        if keys[INPUT.KEY_ACCELERATE]:
            Events.on_keypress_accelerate.broadcast()

        if keys[INPUT.KEY_BRAKE]:
            Events.on_keypress_brake.broadcast()

        if keys[INPUT.KEY_TURN_LEFT]:
            Events.on_keypress_turn.broadcast(data=-1)

        if keys[INPUT.KEY_TURN_RIGHT]:
            Events.on_keypress_turn.broadcast(data=1)

    @staticmethod
    def _update_pressed_keys():

        for event in pygame.event.get(pygame.KEYDOWN):

            if event.key == INPUT.KEY_CHECKPOINTS:
                Events.on_keypress_checkpoints.broadcast()

            if event.key == INPUT.KEY_SENSORS:
                Events.on_keypress_sensors.broadcast()
