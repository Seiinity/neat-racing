import pygame

from config import COLOURS
from pathlib import Path
from pygame import Surface, Color, Rect
from pygame.time import Clock
from src.core.utils import draw_outlined_text
from .button import Button


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

        """
        Initialises the track selector.

        Parameters
        ----------
        tracks_directory : str
            The directory to search for track files.
        """

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((1280, 720))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.selected_track: str | None = None

        pygame.display.set_caption("NEAT-ish Racing - Select Track")

        # Loads available track files.
        self._tracks_directory: Path = Path(tracks_directory)
        self._available_tracks: list[str] = self._load_available_tracks()

        # Scrolling state.
        self._scroll_offset: int = 0
        self._max_visible_items: int = 10
        self._item_height: int = 40
        self._list_start_y: int = 150

        # Creates the control buttons.
        self._start_button: Button = Button(
            x=540,
            y=620,
            width=200,
            height=50,
            text="Select",
            colour=COLOURS.BUTTON_DISABLED
        )

        self._back_button: Button = Button(
            x=100,
            y=620,
            width=150,
            height=50,
            text="Back",
            colour=COLOURS.BACKGROUND
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
            The selected track if confirmed, None to continue.
        """

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

            elif event.type == pygame.MOUSEWHEEL:
                self._handle_scroll(event.y)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._handle_click(event.pos)

            # Handles button clicks.
            if self._back_button.handle_event(event):
                self.running = False

            if self._start_button.handle_event(event):
                if self.selected_track is not None:
                    return self.selected_track

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

    def _handle_click(self, pos: tuple[int, int]) -> None:

        """
        Handles mouse clicks on the track list.

        Parameters
        ----------
        pos : tuple[int, int]
            The mouse position.
        """

        x, y = pos

        # Checks if click is within the list area.
        list_rect = Rect(100, self._list_start_y, 1080, self._max_visible_items * self._item_height)

        if not list_rect.collidepoint(x, y):
            return

        # Calculates which item was clicked.
        relative_y = y - self._list_start_y
        item_index = self._scroll_offset + (relative_y // self._item_height)

        if item_index >= len(self._available_tracks):
            return

        self.selected_track = self._available_tracks[item_index]

    def _draw(self) -> None:

        """
        Draws the track selector screen.
        """

        # Dark background.
        self.screen.fill(COLOURS.BACKGROUND)

        # Title.
        draw_outlined_text(
            self.screen,
            "Select Track",
            (640, 50),
            align="center",
            font_size=48
        )

        # Selected track display.
        if self.selected_track:
            track_name = Path(self.selected_track).stem
            display_text = f"Selected: {track_name}"
            text_colour = Color(100, 200, 100)
        else:
            display_text = "No track selected"
            text_colour = Color(150, 150, 150)

        draw_outlined_text(
            self.screen,
            display_text,
            (640, 100),
            align="center",
            font_size=24,
            text_colour=text_colour
        )

        # Draws the track list.
        self._draw_track_list()

        # Draws scroll indicators if needed.
        self._draw_scroll_indicators()

        # Draws buttons.
        self._back_button.draw(self.screen)

        # Only enables start button if a track is selected.
        if self.selected_track:
            self._start_button.colour = COLOURS.BACKGROUND
        else:
            self._start_button.colour = COLOURS.BUTTON_DISABLED

        self._start_button.draw(self.screen)

        # Instructions.
        draw_outlined_text(
            self.screen,
            "Click to select a track - Scroll to see more",
            (640, 580),
            align="center",
            font_size=18,
            text_colour=Color(100, 100, 100)
        )

        pygame.display.flip()

    def _draw_track_list(self) -> None:

        """
        Draws the list of available tracks.
        """

        if not self._available_tracks:

            draw_outlined_text(
                self.screen,
                "No tracks found in game/tracks/raw folder.",
                (640, 300),
                align="center",
                font_size=24,
                text_colour=Color(150, 100, 100)
            )
            return

        # Draws visible items.
        visible_start = self._scroll_offset
        visible_end = min(visible_start + self._max_visible_items, len(self._available_tracks))

        for i, track_index in enumerate(range(visible_start, visible_end)):

            track_path = self._available_tracks[track_index]
            track_name = Path(track_path).stem
            y = self._list_start_y + (i * self._item_height)

            # Draws the item background.
            item_rect = Rect(100, y, 1080, self._item_height - 4)
            is_selected = track_path == self.selected_track

            if is_selected:
                bg_colour = Color(40, 80, 40)
                text_colour = Color(150, 255, 150)
            else:
                bg_colour = Color(40, 40, 50)
                text_colour = Color(200, 200, 200)

            pygame.draw.rect(self.screen, bg_colour, item_rect, border_radius=4)

            # Draws the selection indicator.
            indicator_text = "[X]" if is_selected else "[ ]"

            # Draws the indicator.
            font = pygame.font.Font(None, 24)
            indicator_surface = font.render(indicator_text, True, text_colour)
            self.screen.blit(indicator_surface, (120, y + 10))

            # Draws the track name.
            name_surface = font.render(track_name, True, text_colour)
            self.screen.blit(name_surface, (180, y + 10))

    def _draw_scroll_indicators(self) -> None:

        """
        Draws scroll indicators if there are more items than visible.
        """

        if len(self._available_tracks) <= self._max_visible_items:
            return

        indicator_colour = Color(100, 100, 100)

        # Up arrow.
        if self._scroll_offset > 0:
            draw_outlined_text(
                self.screen,
                "^ More above ^",
                (640, self._list_start_y - 20),
                align="center",
                font_size=16,
                text_colour=indicator_colour
            )

        # Down arrow.
        max_scroll = len(self._available_tracks) - self._max_visible_items

        if self._scroll_offset < max_scroll:
            bottom_y = self._list_start_y + (self._max_visible_items * self._item_height) + 5
            draw_outlined_text(
                self.screen,
                "v More below v",
                (640, bottom_y),
                align="center",
                font_size=16,
                text_colour=indicator_colour
            )
