from pygame import Color, Surface, Rect
from pygame.font import Font
from pytmx import TiledElement, TiledMap
from config import COLOURS, FONTS


def draw_outlined_text(
        screen: Surface,
        text: str,
        pos: tuple[int, int],
        text_colour: Color = COLOURS.TEXT_MAIN,
        outline_colour: Color = COLOURS.BACKGROUND,
        outline_thickness: int = 2,
        font_size: int = FONTS.SIZE_NORMAL,
        align: str = "centre"
) -> None:

    """
    Draws text with an outline at a given position.

    Parameters
    ----------
    screen : Surface
        The surface to draw on.
    text : str
        The text to draw.
    pos : tuple[int, int]
        The centre position to draw the text at.
    text_colour : Color, optional
        The main text color (defaults to white).
    outline_colour : Color, optional
        The outline color (defaults to black).
    outline_thickness : int, optional
        Thickness of the outline in pixels (defaults to 2).
    font_size : int, optional
        Font size to use.
    align : str, optional
        The alignment of the text.
    """

    # Default font.
    font: Font = Font(FONTS.PATH, font_size)

    # Renders the surfaces.
    outline_surf: Surface = font.render(text, True, outline_colour)
    text_surf: Surface = font.render(text, True, text_colour)

    # Defines a get_rect function based on alignment.
    def get_rect(surf: Surface, position: tuple[int, int]) -> Rect:
        if align == "left":
            return surf.get_rect(topleft=position)
        return surf.get_rect(center=position)

    # Draws the outline in 8 directions.
    for dx in [-outline_thickness, 0, outline_thickness]:
        for dy in [-outline_thickness, 0, outline_thickness]:
            if dx != 0 or dy != 0:
                screen.blit(outline_surf, get_rect(outline_surf, (pos[0] + dx, pos[1] + dy)))

    # Draws the main text.
    screen.blit(text_surf, get_rect(text_surf, pos))


def get_tiled_layer(tmx_data: TiledMap, layer_name: str) -> TiledElement | None:

    """
    Retrieves a layer from a Tiled map by its name.

    Parameters
    ----------
    tmx_data : TiledMap
        The Tiled map to get the layer from.
    layer_name : str
        The name of the layer to retrieve.

    Returns
    -------
    TiledElement | None
        The layer retrieved, or ``None`` if the layer doesn't exist.
    """

    # There's a bug with pytmx's code. This is supposed to return an int,
    # but directly returns a layer instead. This is because the dictionary
    # that is supposed to store layer names their indices actually stores
    # layer names and layers instead.
    layer: TiledElement = tmx_data.get_layer_by_name(layer_name)  # type: ignore

    if layer is None or layer == -1:
        return None

    return layer
