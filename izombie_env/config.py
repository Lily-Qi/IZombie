from enum import Enum


class GameStatus(Enum):
    CONTINUE = 0
    WIN = 1
    LOSE = 2
    TIMEUP = 3


N_LANES = 5  # Height
LANE_LENGTH = 9  # Width
P_LANE_LENGTH = 4  # Plant area width
N_PLANT_TYPE = 4
N_ZOMBIE_TYPE = 3
Z_LANE_LENGTH = LANE_LENGTH - P_LANE_LENGTH
P_MAX_HP = 300
Z_MAX_HP = 80 * 20
SUN_MAX = 1950
ZOMBIE_TYPE_START = 85
ZOMBIE_TYPE_END = 128

ACTION_SIZE = N_ZOMBIE_TYPE * N_LANES + 1
STATE_SIZE = (
    N_LANES * P_LANE_LENGTH  # plant hp
    + N_LANES * P_LANE_LENGTH  # plant type
    + N_LANES * LANE_LENGTH  # zombie hp
    + N_LANES * LANE_LENGTH  # zombie type
    + 1  # sun
    + 5  # brain status
)
