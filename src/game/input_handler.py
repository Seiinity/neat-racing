import pygame
from pygame.key import ScancodeWrapper

from config import INPUT
from src.core import Car, Events


class InputHandler:

    """
    Class which handles player inputs.

    Methods
    -------
    fixed_update() -> None (static)
        Broadcasts events based on the keys pressed by the player.
    """

    @staticmethod
    def fixed_update(player_car: Car) -> None:

        InputHandler._update_held_keys(player_car)
        InputHandler._update_pressed_keys()

    @staticmethod
    def _update_held_keys(player_car: Car):

        keys: ScancodeWrapper = pygame.key.get_pressed()

        if keys[INPUT.KEY_ACCELERATE]:
            Events.on_keypress_accelerate.broadcast(data=player_car)

        if keys[INPUT.KEY_BRAKE]:
            Events.on_keypress_brake.broadcast(data=player_car)

        if keys[INPUT.KEY_TURN_LEFT]:
            Events.on_keypress_turn.broadcast(data=(player_car, -1))

        if keys[INPUT.KEY_TURN_RIGHT]:
            Events.on_keypress_turn.broadcast(data=(player_car, 1))

    @staticmethod
    def _update_pressed_keys():

        for event in pygame.event.get(pygame.KEYDOWN):

            if event.key == INPUT.KEY_CHECKPOINTS:
                Events.on_keypress_checkpoints.broadcast()

            if event.key == INPUT.KEY_SENSORS:
                Events.on_keypress_sensors.broadcast()
