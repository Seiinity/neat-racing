import pygame

from src.training import TrainingLoop
from src.io import GenomeIO
from src.game import GameLoop
from src.ui import MainMenu, TrackSelector, GenomeSelector


def main():

    while True:

        menu = MainMenu()
        selected_mode = menu.run()

        # Exits if no mode was selected (window closed or quit pressed).
        if selected_mode is None:
            break

        # Selects a track.
        track_selector = TrackSelector()
        selected_track = track_selector.run()

        # Returns to menu if track selection was cancelled.
        if selected_track is None:
            continue

        if selected_mode == 'train':

            training = TrainingLoop(
                track_path=selected_track,
                save_interval=10,
                save_best=True
            )
            training.run()

            print("Saving top 5 genomes...")
            GenomeIO.save_best_genomes(
                training.genetic_algorithm,
                num_best=5,
                directory='./data/genomes'
            )
            print("Training complete.")

        elif selected_mode == 'play':

            selector = GenomeSelector('./data/genomes')
            selected_genomes = selector.run()

            # Returns to menu if selection was cancelled.
            if selected_genomes is None or len(selected_genomes) == 0:
                continue

            loop = GameLoop(
                track_path=selected_track,
                genome_paths=selected_genomes
            )
            loop.run()

    pygame.quit()


if __name__ == "__main__":
    main()
