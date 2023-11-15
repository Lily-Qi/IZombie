from enum import Enum


class GameStatus(Enum):
    CONTINUE = 0
    WIN = 1
    LOSE = 2
    TIMEUP = 3


N_LANES = 5  # Height
LANE_LENGTH = 9  # Width
P_LANE_LENGTH = 4
N_PLANT_TYPE = 4
N_ZOMBIE_TYPE = 3
SUN_MAX = 1950

# action
ACTION_SIZE = N_ZOMBIE_TYPE * N_LANES + 1

# state
NUM_ZOMBIES = 39
NUM_PLANTS = 20
ZOMBIE_SIZE = 6
PLANT_SIZE = 4
BRAIN_BASE = NUM_ZOMBIES * ZOMBIE_SIZE + NUM_PLANTS * PLANT_SIZE + 1  # extra 1 for sun
BRAIN_SIZE = 5
STATE_SIZE = BRAIN_BASE + BRAIN_SIZE
