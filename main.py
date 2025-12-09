from algorithm.training_loop import TrainingLoop
from algorithm.genome_io import GenomeIO
from game.game_loop import GameLoop


def main():

    while True:

        print("\nSelect mode:")
        print("  1 - Train AI using genetic algorithms")
        print("  2 - Train AI continuing from best")
        print("  3 - Play manually")
        choice = input("Enter mode (1 or 2): ").strip()

        if choice == '1':

            print("\n=== AI Training Mode ===\n")

            training = TrainingLoop()
            training.run()

            print("\nSaving top 5 genomes...")
            GenomeIO.save_best_genomes(
                training.genetic_algorithm,
                num_best=5,
                directory='./saved_genomes'
            )
            print("\nTraining complete!")
            break

        if choice == '2':

            print("\n=== AI Training Mode - From Best ===\n")

            training = TrainingLoop(GenomeIO.load_genome('./saved_genomes/genome_gen2_rank1.pkl'))
            training.run()

            print("\nSaving top 5 genomes...")
            GenomeIO.save_best_genomes(
                training.genetic_algorithm,
                num_best=5,
                directory='./saved_genomes'
            )
            print("\nTraining complete!")
            break

        elif choice == '3':

            print("\n=== Manual Play Mode ===\n")
            game_loop = GameLoop()
            game_loop.run()
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
