import pygame

from src.training import TrainingLoop
from src.io import GenomeIO
from src.game import GameLoop
from src.ui import MainMenu, TrackSelector, GenomeSelector


def main():
    while True:

        # Main menu.
        menu = MainMenu()
        selected_mode = menu.run()

        # Exits if window closed or quit pressed.
        if selected_mode is None or selected_mode == 'QUIT':
            break

        # Selects a track.
        track_selector = TrackSelector()
        selected_track = track_selector.run()

        # Quits if X was clicked.
        if selected_track == 'QUIT':
            break

        # Returns to menu if track selection was cancelled.
        if selected_track is None:
            continue

        if selected_mode == 'train':

            training = TrainingLoop(
                track_path=selected_track,
                save_interval=10,
                save_best=True
            )
            result = training.run()

            # Quits if X was clicked during training
            if result == 'QUIT':
                break

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

            # Quits if X was clicked.
            if selected_genomes == 'QUIT':
                break

            # Returns to menu if selection was cancelled.
            if selected_genomes is None or len(selected_genomes) == 0:
                continue

            loop = GameLoop(
                track_path=selected_track,
                genome_paths=selected_genomes
            )

            result = loop.run()

            # Quits if X was clicked during gameplay
            if result == 'QUIT':
                break

    pygame.quit()


if __name__ == "__main__":
    main()
