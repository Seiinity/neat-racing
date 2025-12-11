import pygame
import random

from config import COLOURS
from pathlib import Path
from pygame import Surface, Color, Rect
from pygame.time import Clock
from src.core.utils import draw_outlined_text
from .button import Button


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

        self.screen: Surface = pygame.display.set_mode((1280, 720))
        self.clock: Clock = Clock()
        self.running: bool = True
        self.selected_genomes: list[str] = []

        pygame.display.set_caption("NEAT-ish Racing - Select Opponents")

        # Loads available genome files.
        self._genomes_directory: Path = Path(genomes_directory)
        self._available_genomes: list[str] = self._load_available_genomes()

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
            text="Start Race",
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

        self._random_button: Button = Button(
            x=880,
            y=620,
            width=150,
            height=50,
            text="Random",
            colour=COLOURS.BACKGROUND
        )

        self._clear_button: Button = Button(
            x=1030,
            y=620,
            width=150,
            height=50,
            text="Clear",
            colour=COLOURS.BACKGROUND
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

            if self._clear_button.handle_event(event):
                self.selected_genomes.clear()

            if self._random_button.handle_event(event):
                self._select_random()

            if self._start_button.handle_event(event):
                if len(self.selected_genomes) > 0:
                    return self.selected_genomes

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

    def _handle_click(self, pos: tuple[int, int]) -> None:

        """
        Handles mouse clicks on the genome list.

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

        if item_index >= len(self._available_genomes):
            return

        genome_path = self._available_genomes[item_index]

        # Toggles selection.
        if genome_path in self.selected_genomes:
            self.selected_genomes.remove(genome_path)
        elif len(self.selected_genomes) < self.MAX_SELECTIONS:
            self.selected_genomes.append(genome_path)

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

        # Dark background.
        self.screen.fill(COLOURS.BACKGROUND)

        # Title.
        draw_outlined_text(
            self.screen,
            "Select Opponents",
            (640, 50),
            align="center",
            font_size=48
        )

        # Selection count.
        count_colour = Color(100, 200, 100) if self.selected_genomes else Color(150, 150, 150)
        draw_outlined_text(
            self.screen,
            f"Selected: {len(self.selected_genomes)}/{self.MAX_SELECTIONS}",
            (640, 100),
            align="center",
            font_size=24,
            text_colour=count_colour
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
        if self.selected_genomes:
            self._start_button.colour = COLOURS.BACKGROUND
        else:
            self._start_button.colour = COLOURS.BUTTON_DISABLED

        self._start_button.draw(self.screen)

        # Instructions.
        draw_outlined_text(
            self.screen,
            "Click to select/deselect - Scroll to see more",
            (640, 580),
            align="center",
            font_size=18,
            text_colour=Color(100, 100, 100)
        )

        pygame.display.flip()

    def _draw_genome_list(self) -> None:

        """
        Draws the list of available genomes.
        """

        if not self._available_genomes:

            draw_outlined_text(
                self.screen,
                "No genomes found in data/genomes folder.",
                (640, 300),
                align="center",
                font_size=24,
                text_colour=Color(150, 100, 100)
            )
            return

        # Draws visible items.
        visible_start = self._scroll_offset
        visible_end = min(visible_start + self._max_visible_items, len(self._available_genomes))

        for i, genome_index in enumerate(range(visible_start, visible_end)):

            genome_path = self._available_genomes[genome_index]
            genome_name = Path(genome_path).name
            y = self._list_start_y + (i * self._item_height)

            # Draws the item background.
            item_rect = Rect(100, y, 1080, self._item_height - 4)
            is_selected = genome_path in self.selected_genomes

            if is_selected:
                bg_colour = Color(40, 80, 40)
                text_colour = Color(150, 255, 150)
            else:
                bg_colour = Color(40, 40, 50)
                text_colour = Color(200, 200, 200)

            pygame.draw.rect(self.screen, bg_colour, item_rect, border_radius=4)

            # Draws the selection indicator.
            if is_selected:
                selection_index = self.selected_genomes.index(genome_path) + 1
                indicator_text = f"[{selection_index}]"
            else:
                indicator_text = "[ ]"

            # Draws the indicator.
            font = pygame.font.Font(None, 24)
            indicator_surface = font.render(indicator_text, True, text_colour)
            self.screen.blit(indicator_surface, (120, y + 10))

            # Draws the genome name.
            name_surface = font.render(genome_name, True, text_colour)
            self.screen.blit(name_surface, (180, y + 10))

    def _draw_scroll_indicators(self) -> None:

        """
        Draws scroll indicators if there are more items than visible.
        """

        if len(self._available_genomes) <= self._max_visible_items:
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
        max_scroll = len(self._available_genomes) - self._max_visible_items

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
