from numpy.random import Generator, default_rng
from config import RANDOM_SEED

rng: Generator = default_rng(seed=RANDOM_SEED)