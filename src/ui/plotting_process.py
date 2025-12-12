import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any
from queue import Empty
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

    while True:

        # Checks for new data.
        try:
            data: dict[str, Any] = queue.get(timeout=0.1)
        except Empty:
            plt.pause(0.01)
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

        # Clears and redraws.
        for ax in axes.flat:
            ax.clear()

        # Fitness over time.
        axes[0, 0].plot(generations, best_fitness, label='Best', color='green')
        axes[0, 0].plot(generations, avg_fitness, label='Avg', color='blue')
        axes[0, 0].plot(generations, worst_fitness, label='Worst', color='red')
        axes[0, 0].set_title("Fitness Over Generations")
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Fitness")
        axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[0, 0].legend()

        # Fitness distribution (current generation).
        if 'fitness_distribution' in data:

            sns.histplot(data['fitness_distribution'], ax=axes[0, 1], kde=True)
            axes[0, 1].set_title(f"Fitness Distribution (Gen {current_gen})")
            axes[0, 1].set_xlabel("Fitness")
            axes[0, 1].set_ylabel("Count")

        # Checkpoints reached (current generation, lap 0 only).
        if 'checkpoints' in data and 'laps' in data:

            filtered_checkpoints = [
                cp for cp, lap in zip(data['checkpoints'], data['laps']) if lap == 0
            ]

            if filtered_checkpoints:
                sns.histplot(filtered_checkpoints, ax=axes[0, 2], discrete=True)

            axes[0, 2].set_title(f"Checkpoints Reached (Gen {current_gen}, Lap 0)")
            axes[0, 2].set_xlabel("Checkpoint")
            axes[0, 2].set_ylabel("Count")
            axes[0, 2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Laps completed (current generation).
        if 'laps' in data:

            sns.histplot(data['laps'], ax=axes[1, 0], discrete=True)
            axes[1, 0].set_title(f"Laps Completed (Gen {current_gen})")
            axes[1, 0].set_xlabel("Laps")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Survival times (current generation).
        if 'survival_times' in data:

            sns.histplot(data['survival_times'], ax=axes[1, 1], kde=True)
            axes[1, 1].set_title(f"Survival Times (Gen {current_gen})")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Count")

        # Death position heatmap with track overlay.
        if track_img is not None:

            track_height, track_width = track_img.shape[:2]
            axes[1, 2].imshow(track_img, extent=[0, track_width, track_height, 0], aspect='equal')

            if 'death_positions' in data and len(data['death_positions']) > 1:

                death_x: list[float] = [pos[0] for pos in data['death_positions']]
                death_y: list[float] = [pos[1] for pos in data['death_positions']]

                sns.kdeplot(
                    x=death_x, y=death_y, ax=axes[1, 2],
                    fill=True, cmap='Reds', alpha=0.3,
                )

            axes[1, 2].set_xlim(0, track_width)
            axes[1, 2].set_ylim(track_height, 0)
            axes[1, 2].grid(alpha=0.1)

        axes[1, 2].set_title("Death Heatmap (All Generations)")
        axes[1, 2].set_xlabel("X")
        axes[1, 2].set_ylabel("Y")

        fig.tight_layout()
        plt.pause(0.01)
