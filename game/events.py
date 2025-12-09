from game.core.event import Event


class Events:

    """
    Central repository for all events.

    Events can be subscribed to using ``Event.add_listener()``.
    To unsubscribe, ``Event.remove_listener()`` is used.

    Broadcast events by calling ``Event.broadcast()``.
    """

    # User input events.
    on_keypress_accelerate: Event = Event()
    on_keypress_brake: Event = Event()
    on_keypress_turn: Event = Event()

    # Debug input events.
    on_keypress_checkpoints: Event = Event()
    on_keypress_sensors: Event = Event()

    # Game events.
    on_car_moved: Event = Event()
    on_car_collided: Event = Event()
    on_car_die: Event = Event()
    on_checkpoint_hit: Event = Event()
    on_finish_line_crossed: Event = Event()
    on_lap_completed: Event = Event()
