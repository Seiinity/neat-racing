from collections.abc import Callable

class Event:

    """
    Unity-esque event class.
    Allows for the addition and removal of listeners. Broadcasting
    an event calls any active listeners of that event.

    Methods
    -------
    add_listener(self, listener) -> None
        Adds a function to the event's listeners.
    remove_listener(self, listener) -> None
        Removes a function from the event's listeners.
    broadcast(self, **data) -> None
        Broadcasts an event to all active listeners, with optional data.
    """

    def __init__(self) -> None:

        self.listeners: list[Callable[..., None]] = []

    def add_listener(self, listener) -> None:

        """
        Adds a function to the event's listeners.

        Parameters
        ----------
        listener : Callable[..., None]
            The function to be added to the event's listeners.
        """

        if listener not in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener) -> None:

        """
        Removes a function from the event's listeners.

        Parameters
        ----------
        listener : Callable[..., None]
            The function to be removed from the event's listeners.
        """

        if listener in self.listeners:
            self.listeners.remove(listener)

    def broadcast(self, **data) -> None:

        """
        Broadcasts an event to all active listeners, with optional data.

        Parameters
        ----------
        data
            The data to be passed to the event's listeners.
        """

        for listener in self.listeners:
            listener(**data)