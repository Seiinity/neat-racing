import pygame
import random

from config import COLOURS, FONTS, GAME
from pathlib import Path
from pygame import Surface
from pygame.time import Clock
from src.core.utils import draw_outlined_text
from .button import Button
from .list_item import ListItem


class GenomeSelector:

    """
    Genome selection screen for choosing AI opponents.

    Attributes
    ----------
    screen : Surface
        The Pygame display surface.
    clock : Clock
        The Pygame clock for timing.
    running : bool
        Whether the selector is currently running.
    selected_genomes : list[str]
        A list of selected genome file paths.
    """

    MAX_SELECTIONS: int = 5

    def __init__(self, genomes_directory: str = './data/genomes') -> None:

        """
        Initialises the genome selector.

        Parameters
        ----------
        genomes_directory : str
            The directory to search for genome files.
        """

        pygame.init()

        self.screen: Surface = pygame.display.set_mode((GAME.SCREEN_WIDTH, GAME.SCREEN_HEIGHT))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.selected_genomes: list[str] = []

        pygame.display.set_caption("NEAT-ish Racing - Select Opponents")

        # Resets the cursor.
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        # Loads available genome files.
        self._genomes_directory: Path = Path(genomes_directory)
        self._available_genomes: list[str] = self._load_available_genomes()

        # Scrolling state.
        self._scroll_offset: int = 0
        self._last_scroll_offset: int = 0
        self._max_visible_items: int = 8
        self._item_height: int = 40
        self._list_start_y: int = 180
        self._genome_items: list[ListItem] = []

        button_width: int = 100
        button_height: int = 40

        # Creates the control buttons.
        self._start_button: Button = Button(
            x=GAME.SCREEN_WIDTH // 2 - 75,
            y=620,
            width=150,
            height=button_height,
            text="Start Race",
            disabled=True
        )

        self._back_button: Button = Button(
            x=100,
            y=620,
            width=button_width,
            height=button_height,
            text="Back"
        )

        self._random_button: Button = Button(
            x=925,
            y=620,
            width=button_width,
            height=button_height,
            text="Random"
        )

        self._clear_button: Button = Button(
            x=1030,
            y=620,
            width=button_width,
            height=button_height,
            text="Clear"
        )

    def _load_available_genomes(self) -> list[str]:

        """
        Loads available genome files from the directory.

        Returns
        -------
        list[str]
            A list of genome file paths.
        """

        if not self._genomes_directory.exists():
            return []

        genome_files = list(self._genomes_directory.glob('*.pkl'))
        genome_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return [str(path) for path in genome_files]

    def run(self) -> list[str] | None:

        """
        Runs the genome selector loop.

        Returns
        -------
        list[str] | None
            A list of selected genome paths, or None if cancelled.
        """

        while self.running:

            result = self._process_events()

            if result is not None:
                return result

            self._draw()
            self.clock.tick(60)

        return None

    def _process_events(self) -> list[str] | None:

        """
        Processes all pending Pygame events.

        Returns
        -------
        list[str] | None
            The selected genomes if start was pressed, None to continue.
        """

        # Only updates items if the scroll state changed or items don't exist yet.
        if self._scroll_offset != self._last_scroll_offset or not self._genome_items:
            self._update_genome_items()
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

            if self._clear_button.handle_event(event):

                self.selected_genomes.clear()

                # Updates the selection state for all items.
                for item in self._genome_items:
                    item.is_selected = False
                    item.selection_index = None

            if self._random_button.handle_event(event):

                self._select_random()

                # Update items after random selection.

                for item in self._genome_items:

                    if item.data in self.selected_genomes:
                        item.is_selected = True
                        item.selection_index = self.selected_genomes.index(item.data) + 1
                    else:
                        item.is_selected = False
                        item.selection_index = None

            if self._start_button.handle_event(event):
                if len(self.selected_genomes) > 0:
                    return self.selected_genomes

            # Handles list item clicks.
            for item in self._genome_items:

                if item.handle_event(event):
                    genome_path = item.data

                    # Toggle selections
                    if genome_path in self.selected_genomes:
                        self.selected_genomes.remove(genome_path)
                    elif len(self.selected_genomes) < self.MAX_SELECTIONS:
                        self.selected_genomes.append(genome_path)

                    # Update all items' selection states
                    for it in self._genome_items:

                        if it.data in self.selected_genomes:
                            it.is_selected = True
                            it.selection_index = self.selected_genomes.index(it.data) + 1
                        else:
                            it.is_selected = False
                            it.selection_index = None

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

        max_scroll = max(0, len(self._available_genomes) - self._max_visible_items)
        self._scroll_offset = max(0, min(max_scroll, self._scroll_offset - direction))

    def _select_random(self) -> None:

        """
        Selects up to 5 random genomes from the available options.
        """

        if not self._available_genomes:
            return

        self.selected_genomes.clear()

        # Selects up to MAX_SELECTIONS random genomes.
        num_to_select = min(self.MAX_SELECTIONS, len(self._available_genomes))
        self.selected_genomes = random.sample(self._available_genomes, num_to_select)

    def _draw(self) -> None:

        """
        Draws the genome selector screen.
        """

        # Background.
        self.screen.fill(COLOURS.BACKGROUND)

        # Title.
        draw_outlined_text(
            self.screen,
            "Select Opponents",
            (GAME.SCREEN_WIDTH // 2, 80),
            font_size=FONTS.SIZE_LARGE
        )

        # Selection count.
        draw_outlined_text(
            self.screen,
            f"Selected: {len(self.selected_genomes)}/{self.MAX_SELECTIONS}",
            (GAME.SCREEN_WIDTH // 2, 110),
            font_size=FONTS.SIZE_NORMAL,
            text_colour=COLOURS.TEXT_SECONDARY
        )

        # Draws the genome list.
        self._draw_genome_list()

        # Draws scroll indicators if needed.
        self._draw_scroll_indicators()

        # Draws buttons.
        self._back_button.draw(self.screen)
        self._random_button.draw(self.screen)
        self._clear_button.draw(self.screen)

        # Only enables start button if at least one genome is selected.
        self._start_button.disabled = not self.selected_genomes
        self._start_button.draw(self.screen)

        # Instructions.
        draw_outlined_text(
            self.screen,
            "Click to select/deselect. Scroll to see more.",
            (GAME.SCREEN_WIDTH // 2, 580),
            font_size=FONTS.SIZE_NORMAL,
            text_colour=COLOURS.TEXT_SECONDARY
        )

        pygame.display.flip()

    def _update_genome_items(self) -> None:

        """
        Updates the list items based on current scroll position.
        """

        self._genome_items.clear()
        visible_start = self._scroll_offset
        visible_end = min(visible_start + self._max_visible_items, len(self._available_genomes))

        item_width = 600
        item_position = GAME.SCREEN_WIDTH // 2 - item_width // 2

        for i, genome_index in enumerate(range(visible_start, visible_end)):

            genome_path = self._available_genomes[genome_index]
            genome_name = Path(genome_path).stem
            y = self._list_start_y + (i * self._item_height)

            item = ListItem(
                item_position,
                y,
                item_width,
                self._item_height - 4,
                genome_name,
                data=genome_path
            )

            # Sets the selection state and index.
            if genome_path in self.selected_genomes:

                item.is_selected = True
                item.selection_index = self.selected_genomes.index(genome_path) + 1

            self._genome_items.append(item)

    def _draw_genome_list(self) -> None:

        """
        Draws the list of available genomes.
        """

        if not self._available_genomes:
            draw_outlined_text(
                self.screen,
                "No genomes found in data/genomes folder.",
                (GAME.SCREEN_WIDTH // 2, 300),
                font_size=FONTS.SIZE_NORMAL,
                text_colour=COLOURS.TEXT_ERROR
            )
            return

        # Draws all items.
        for item in self._genome_items:
            item.draw(self.screen)

    def _draw_scroll_indicators(self) -> None:

        """
        Draws scroll indicators if there are more items than visible.
        """

        if len(self._available_genomes) <= self._max_visible_items:
            return

        indicator_colour = COLOURS.TEXT_SECONDARY

        # Up arrow.
        if self._scroll_offset > 0:
            draw_outlined_text(
                self.screen,
                "▲",
                (GAME.SCREEN_WIDTH // 2, self._list_start_y - 20),
                align="center",
                font_size=FONTS.SIZE_NORMAL,
                text_colour=indicator_colour
            )

        # Down arrow.
        max_scroll = len(self._available_genomes) - self._max_visible_items

        if self._scroll_offset < max_scroll:
            bottom_y = self._list_start_y + (self._max_visible_items * self._item_height) + 5
            draw_outlined_text(
                self.screen,
                "▼",
                (GAME.SCREEN_WIDTH // 2, bottom_y + 10),
                align="center",
                font_size=FONTS.SIZE_NORMAL,
                text_colour=indicator_colour
            )
