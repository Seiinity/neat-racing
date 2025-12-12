import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any
from queue import Empty
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.colorbar import Colorbar
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
from numpy.typing import NDArray


def plotting_process(queue: mp.Queue):

    """
    Runs in a separate process, receives data via queue.
    """

    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#1a1a1a",
        "figure.facecolor": "#121212",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#333333"
    })

    # Turns on the interactive mode.
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Track background (received with first data packet).
    track_img: NDArray | None = None

    # Colour bar reference for cleanup.
    heatmap_cbar: Colorbar | None = None

    # Stores the original positions to restore the grid view.
    original_positions: dict[Axes, Bbox] = {ax: ax.get_position() for ax in axes.flat}
    zoomed_ax: Axes | None = None

    def on_click(event: Event) -> None:

        if not isinstance(event, MouseEvent):
            return None

        nonlocal zoomed_ax

        if event.inaxes is None:
            return None

        # If the view is already zoomed, restores the grid view.
        if zoomed_ax is not None:

            for ax in axes.flat:
                ax.set_visible(True)
                ax.set_position(original_positions[ax])

            zoomed_ax = None
            fig.canvas.draw_idle()
            return None

        # Zooms into clicked subplot.
        clicked_ax: Axes = event.inaxes
        if clicked_ax in axes.flat:

            for ax in axes.flat:
                if ax != clicked_ax:
                    ax.set_visible(False)

            clicked_ax.set_position((0.1, 0.1, 0.8, 0.8))
            zoomed_ax = clicked_ax
            fig.canvas.draw_idle()

        return None

    # Links the defined function to matplotlib's click event.
    fig.canvas.mpl_connect('button_press_event', on_click)

    while True:

        # Checks for new data.
        try:
            data: dict[str, Any] = queue.get(timeout=0.1)
        except Empty:
            fig.canvas.flush_events()
            continue

        # Shutdown.
        if data is None:
            plt.close()
            break

        # Stores track background if included.
        if 'track_bg' in data:
            track_img = data['track_bg']

        # Extracts history and current gen data.
        generations: list = data['generations']
        best_fitness: list = data['best_fitness']
        avg_fitness: list = data['avg_fitness']
        worst_fitness: list = data['worst_fitness']
        current_gen: int = data['current_gen']

        # Updates title with current generation.
        fig.suptitle(f"Training Progress — Generation {current_gen}")

        # Removes old colour bar.
        if heatmap_cbar is not None:
            heatmap_cbar.remove()
            heatmap_cbar = None

        # Clears and redraws.
        for ax in axes.flat:
            ax.clear()

        # Restores zoom state after clearing.
        if zoomed_ax is not None:
            for ax in axes.flat:
                if ax != zoomed_ax:
                    ax.set_visible(False)
                else:
                    ax.set_position((0.1, 0.1, 0.8, 0.8))

        # Fitness over time.
        sns.lineplot(x=generations, y=best_fitness, label='Best', color='mediumseagreen', ax=axes[0, 0])
        sns.lineplot(x=generations, y=avg_fitness, label='Average', color='deepskyblue', ax=axes[0, 0])
        sns.lineplot(x=generations, y=worst_fitness, label='Worst', color='tomato', ax=axes[0, 0])
        axes[0, 0].set_title("Fitness Over Generations")
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Fitness")
        axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[0, 0].legend()

        # Fitness distribution (current generation).
        if 'fitness_distribution' in data:

            sns.histplot(data['fitness_distribution'], ax=axes[0, 1], kde=True, color='mediumpurple')
            axes[0, 1].set_title(f"Fitness Distribution (Gen {current_gen})")
            axes[0, 1].set_xlabel("Fitness")
            axes[0, 1].set_ylabel("Count")

        # Checkpoints reached (current generation, lap 0 only).
        if 'checkpoints' in data and 'laps' in data:

            filtered = [cp for cp, lap in zip(data['checkpoints'], data['laps']) if lap == 0]

            # If everyone has 0 checkpoints, shows a blank message.
            if len(filtered) == 0 or all(cp == 0 for cp in filtered):

                _draw_empty_message(axes[0, 2], "No car has crossed a checkpoint.")

            else:

                sns.histplot(filtered, ax=axes[0, 2], discrete=True, color='darkorange')
                axes[0, 2].set_xlabel("Checkpoint")
                axes[0, 2].set_ylabel("Count")
                axes[0, 2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            axes[0, 2].set_title(f"Checkpoints Reached (Gen {current_gen}, Lap 0)")

        # Laps completed (current generation).
        if 'laps' in data:
            laps = data['laps']

            # If everyone has 0 laps, shows a blank message.
            if len(laps) == 0 or all(lp == 0 for lp in laps):

                _draw_empty_message(axes[1, 0], "No car has completed a lap.")

            else:

                sns.histplot(laps, ax=axes[1, 0], discrete=True, color='teal')
                axes[1, 0].set_xlabel("Laps")
                axes[1, 0].set_ylabel("Count")
                axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            axes[1, 0].set_title(f"Laps Completed (Gen {current_gen})")

        # Survival times (current generation).
        if 'survival_times' in data:

            sns.histplot(data['survival_times'], ax=axes[1, 1], kde=True, color='coral')
            axes[1, 1].set_title(f"Survival Times (Gen {current_gen})")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Count")

        # Death position heatmap with track overlay.
        heatmap_ax = axes[1, 2]
        is_heatmap_zoomed = (zoomed_ax is heatmap_ax)

        if track_img is not None:

            track_height, track_width = track_img.shape[:2]

            # Use 'auto' aspect when zoomed to allow proper scaling.
            aspect: str = 'auto' if is_heatmap_zoomed else 'equal'
            heatmap_ax.imshow(track_img, extent=[0, track_width, track_height, 0], aspect=aspect)

            if 'death_positions' in data and len(data['death_positions']) > 1:

                death_x: list[float] = [pos[0] for pos in data['death_positions']]
                death_y: list[float] = [pos[1] for pos in data['death_positions']]

                bin_size: int = 20
                bins_x: int = int(track_width / bin_size)
                bins_y: int = int(track_height / bin_size)

                _, _, _, mesh = heatmap_ax.hist2d(
                    death_x, death_y,
                    bins=[bins_x, bins_y],
                    cmap='Reds',
                    alpha=0.4,
                    range=[[0, track_width], [0, track_height]]
                )

                # Only shows the colour bar if not zoomed into a different plot.
                if zoomed_ax is None or is_heatmap_zoomed:
                    heatmap_cbar = fig.colorbar(mesh, ax=heatmap_ax)
                    heatmap_cbar.set_label('Deaths')

            heatmap_ax.set_xlim(0, track_width)
            heatmap_ax.set_ylim(track_height, 0)
            heatmap_ax.grid(alpha=0.1)

        heatmap_ax.set_title("Death Heatmap (All Generations)")
        heatmap_ax.set_xlabel("X")
        heatmap_ax.set_ylabel("Y")

        if zoomed_ax is None:

            fig.tight_layout()

        else:

            for ax in axes.flat:

                if ax != zoomed_ax:
                    ax.set_visible(False)
                else:
                    ax.set_position((0.1, 0.1, 0.8, 0.8))

            # Positions the colour bar when the heatmap is zoomed.
            if is_heatmap_zoomed and heatmap_cbar is not None:
                heatmap_cbar.ax.set_visible(True)
                heatmap_cbar.ax.set_position((0.87, 0.1, 0.03, 0.8))
            elif heatmap_cbar is not None:
                heatmap_cbar.ax.set_visible(False)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()


def _draw_empty_message(ax: Axes, message: str) -> None:

    """
    Draws a message when there is no relevant data.

    Parameters
    ----------
    ax : Axes
        The axes on which to draw the message.
    message : str
        The message to draw.
    """

    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", fontsize=12, color="white",
        transform=ax.transAxes
    )
