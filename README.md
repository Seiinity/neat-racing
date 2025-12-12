# NEAT-ish Racing
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pygame](https://img.shields.io/badge/Pygame-2.5.0-5AA816.svg?style=flat)](https://www.pygame.org/news)

A neural network racing game where AI cars learn to drive using a genetic algorithm inspired by **NEAT (NeuroEvolution of Augmenting Topologies)**.<br/>
Watch cars evolve from crashing into walls to smoothly navigating tracks, then race against the best AI yourself.

![Thumbnail](https://i.imgur.com/DRFXffL.png)

## Features

### Training Mode
- **Genetic Algorithm**: Neural network-based evolution with selection and mutation;
- **Real-time Visualisation**: Toggle between console output and visual training;
- **Live Statistics**: Matplotlib dashboard with fitness graphs, checkpoint distribution, survival times, and death heatmaps;
- **Autosave**: Periodic saving of top-performing genomes.

### Game Mode
- **Player vs AI**: Race against the trained neural networks;
- **Multiple Tracks**: Load custom tracks created in Tiled.

## Installation

```bash
# Clone the repository
git clone https://github.com/seiinity/neat-racing.git
cd neat-racing

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- [Python 3.13+](https://www.python.org/downloads/)
- [Pygame](https://www.pygame.org/download.shtml)
- NumPy
- Matplotlib
- Seaborn
- Shapely
- [PyTMX](https://github.com/bitcraft/pytmx)

## Usage

```bash
python main.py
```

### Main Menu
- **Train**: Start training a new population on a selected track;
- **Play**: Race against saved genomes on a selected track.

### Game Controls

| Key | Action                          |
|-----|---------------------------------|
| W   | Accelerate                      |
| S   | Brake                           |
| A   | Turn left                       |
| D   | Turn right                      |
| 0   | Toggle checkpoint visualisation |
| 1   | Toggle sensor visualisation     |

## Project Structure

```
neat-racing/
├── config/                        # Configuration files.
│   ├── algorithm_config.py        # Genetic algorithm parameters.
│   ├── game_config.py             # Display and physics settings.
│   ├── render_config.py           # Colours and fonts.
│   └── training_config.py         # Training parameters.
├── data/
│   ├── fonts/                     # Custom fonts.
│   ├── genomes/                   # Saved neural network genomes.
│   └── tracks/                    # Track files (.tmx) and images.
├── docs/                          # Documentation
├── src/
│   ├── algorithm/                 # Neural network and genetic algorithm.
│   ├── core/                      # Car, track, and event system.
│   ├── game/                      # Game loop and input handling.
│   ├── io/                        # Genome serialisation.
│   ├── training/                  # Training loop and AI controller.
│   └── ui/                        # Menus, buttons, and plotting.
└── main.py                        # Entry point.
```

## How It Works

### Neural Network

Each car is controlled by a neural network that receives **sensor inputs** (distances to walls) and outputs **control signals** (accelerate, brake, turn left, turn right).

### Genetic Algorithm

1. **Evaluation**: Each genome controls a car; fitness is based on velocity, distance to walls, distance travelled, checkpoints reached, laps completed, and time wasted;
2. **Selection**: Top performers are selected for crossover;
3. **Crossover**: New genomes are created from a mix of two parent genomes;
4. **Mutation**: Random modifications to the topology, weights, or activation functions;
5. **Repeat**: The process continues until the cars master the track.

## Creating Custom Tracks

Tracks are created using [Tiled](https://www.mapeditor.org/):

1. Create a new map with your track image as a background layer named `bg`.
2. Add an object layer named `objects` with:
   - Checkpoints (type: `checkpoint`, property `order`: 0, 1, 2...);
   - Finish line (name: `finish_line`);
   - Start positions (type: `start_pos` for AI, `start_pos_player` for player).
3. Add an object layer named `bounds` with:
   - Outer boundary polygon (name: `outer_bound`);
   - Inner boundary polygon (name: `inner_bound`).
4. Save as `.tmx` in `data/tracks/raw/`.

## Configuration

Key parameters can be adjusted in the `config/` files.<br/>
These are some examples:

### Training (`training_config.py`)

- `POPULATION_SIZE`: Number of cars per generation;
- `MAX_GENERATION_TIME`: Time limit per generation, in seconds;
- `SPEED`: Training speed multiplier (console mode).

### Algorithm (`algorithm_config.py`)

- `MIN_LAYERS`, `MAX_LAYERS`, `MIN_NEURONS`, `MAX_NEURONS`: Genome topology limits;
- `MUTATION_RATE`, `TOPOLOGY_RATE`, `ACTIVATION_RATE`: Probability of mutations.

### Car (`game_config.py`)

- `SENSORS`: Angles for distance sensors;
- `SENSOR_RANGE`: Maximum sensor detection distance;
- `ACCELERATION`, `BRAKE_STRENGTH`, `TURN_SPEED`: Car physics.

## Included Genomes

The repository includes the top 10 genomes from generations 50, 100, 300, and 1000.<br/>
They were trained with the default config values.

## License

MIT License - feel free to use, modify, and distribute.
