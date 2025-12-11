import pygame

from config import COLOURS, FONTS, GAME
from pathlib import Path
from pygame import Surface
from pygame.time import Clock
from src.core.utils import draw_outlined_text
from .button import Button
from .list_item import ListItem


class TrackSelector:

    """
    Track selection screen for choosing a racing track.

    Attributes
    ----------
    screen : Surface
        The Pygame display surface.
    clock : Clock
        The Pygame clock for timing.
    running : bool
        Whether the selector is currently running.
    selected_track : str | None
        The selected track file path, or None if not yet selected.
    """

    def __init__(self, tracks_directory: str = './data/tracks/raw') -> None:

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((GAME.SCREEN_WIDTH, GAME.SCREEN_HEIGHT))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.selected_track: str | None = None
        self._track_items: list[ListItem] = []

        pygame.display.set_caption("NEAT-ish Racing - Select Track")

        # Resets the cursor.
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        # Loads available track files.
        self._tracks_directory: Path = Path(tracks_directory)
        self._available_tracks: list[str] = self._load_available_tracks()

        # Scrolling state.
        self._scroll_offset: int = 0
        self._last_scroll_offset: int = 0
        self._max_visible_items: int = 8
        self._item_height: int = 40
        self._list_start_y: int = 180

        # Creates the control buttons.
        self._start_button: Button = Button(
            x=GAME.SCREEN_WIDTH // 2 - 75,
            y=620,
            width=150,
            height=40,
            text="Continue",
            disabled=True
        )

        self._back_button: Button = Button(
            x=100,
            y=620,
            width=100,
            height=40,
            text="Back"
        )

    def _load_available_tracks(self) -> list[str]:

        """
        Loads available track files from the directory.

        Returns
        -------
        list[str]
            A list of track file paths.
        """

        if not self._tracks_directory.exists():
            return []

        track_files = list(self._tracks_directory.glob('*.tmx'))
        track_files.sort(key=lambda p: p.name)

        return [str(path) for path in track_files]

    def run(self) -> str | None:

        """
        Runs the track selector loop.

        Returns
        -------
        str | None
            The selected track path, or None if cancelled.
        """

        while self.running:

            result = self._process_events()

            if result is not None:
                return result

            self._draw()
            self.clock.tick(60)

        return None

    def _process_events(self) -> str | None:

        """
        Processes all pending Pygame events.

        Returns
        -------
        str | None
            The selected track if confirmed, ``None`` to continue.
        """

        # Only updates items if the scroll state changed or items don't exist yet.
        if self._scroll_offset != self._last_scroll_offset or not self._track_items:
            self._update_track_items()
            self._last_scroll_offset = self._scroll_offset

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

            elif event.type == pygame.MOUSEWHEEL:
                self._handle_scroll(event.y)

            # Handles button clicks.
            if self._back_button.handle_event(event):
                self.running = False

            if self._start_button.handle_event(event):
                if self.selected_track is not None:
                    return self.selected_track

            # Handles list items events.
            for item in self._track_items:
                if item.handle_event(event):

                    self.selected_track = item.data

                    # Updates selection state for all items.
                    for it in self._track_items:
                        it.is_selected = (it.data == self.selected_track)

                    break

        return None

    def _handle_scroll(self, direction: int) -> None:

        """
        Handles mouse wheel scrolling.

        Parameters
        ----------
        direction : int
            The scroll direction (positive for up, negative for down).
        """

        max_scroll = max(0, len(self._available_tracks) - self._max_visible_items)
        self._scroll_offset = max(0, min(max_scroll, self._scroll_offset - direction))

    def _draw(self) -> None:

        """
        Draws the track selector screen.
        """

        # Background.
        self.screen.fill(COLOURS.BACKGROUND)

        # Title.
        draw_outlined_text(
            self.screen,
            "Select Track",
            (GAME.SCREEN_WIDTH // 2, 80),
            font_size=FONTS.SIZE_LARGE
        )

        # Selected track display.
        if self.selected_track:
            track_name = Path(self.selected_track).stem
            display_text = f"Selected: {track_name}"
        else:
            display_text = "No track selected"

        draw_outlined_text(
            self.screen,
            display_text,
            (GAME.SCREEN_WIDTH // 2, 110),
            font_size=FONTS.SIZE_NORMAL,
            text_colour=COLOURS.TEXT_SECONDARY
        )

        # Draws the track list.
        self._draw_track_list()

        # Draws scroll indicators if needed.
        self._draw_scroll_indicators()

        # Draws buttons.
        self._back_button.draw(self.screen)

        # Only enables the start button if a track is selected.
        self._start_button.disabled = not self.selected_track
        self._start_button.draw(self.screen)

        # Instructions.
        draw_outlined_text(
            self.screen,
            "Click to select a track. Scroll to see more.",
            (GAME.SCREEN_WIDTH // 2, 580),
            font_size=FONTS.SIZE_NORMAL,
            text_colour=COLOURS.TEXT_SECONDARY
        )

        pygame.display.flip()

    def _update_track_items(self) -> None:

        """
        Updates the list items based on current scroll position.
        """

        # Hides items if overflowing.
        self._track_items.clear()
        visible_start = self._scroll_offset
        visible_end = min(visible_start + self._max_visible_items, len(self._available_tracks))

        item_width = 600
        item_position = GAME.SCREEN_WIDTH // 2 - item_width // 2

        # Creates a ListItem for each item of the list.
        for i, track_index in enumerate(range(visible_start, visible_end)):

            track_path = self._available_tracks[track_index]
            track_name = Path(track_path).stem
            y = self._list_start_y + (i * self._item_height)

            item = ListItem(
                item_position,
                y,
                item_width,
                self._item_height - 4,
                track_name,
                data=track_path
            )

            item.is_selected = (track_path == self.selected_track)
            self._track_items.append(item)

    def _draw_track_list(self) -> None:

        """
        Draws the list of available tracks.
        """

        if not self._available_tracks:

            draw_outlined_text(
                self.screen,
                "No tracks found in game/tracks/raw folder.",
                (GAME.SCREEN_WIDTH // 2, 300),
                font_size=FONTS.SIZE_NORMAL,
                text_colour=COLOURS.TEXT_ERROR
            )
            return

        # Draws all items.
        for item in self._track_items:
            item.draw(self.screen)

    def _draw_scroll_indicators(self) -> None:

        """
        Draws scroll indicators if there are more items than visible.
        """

        if len(self._available_tracks) <= self._max_visible_items:
            return

        indicator_colour = COLOURS.TEXT_SECONDARY

        # Up arrow.
        if self._scroll_offset > 0:
            draw_outlined_text(
                self.screen,
                "▲",
                (GAME.SCREEN_WIDTH // 2, self._list_start_y - 20),
                font_size=FONTS.SIZE_NORMAL,
                text_colour=indicator_colour
            )

        # Down arrow.
        max_scroll = len(self._available_tracks) - self._max_visible_items

        if self._scroll_offset < max_scroll:
            bottom_y = self._list_start_y + (self._max_visible_items * self._item_height) + 5
            draw_outlined_text(
                self.screen,
                "▼",
                (GAME.SCREEN_WIDTH // 2, bottom_y + 10),
                font_size=FONTS.SIZE_NORMAL,
                text_colour=indicator_colour
            )
