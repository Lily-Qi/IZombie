import numpy as np
import json
import math
from pvzemu import World, SceneType, ZombieType, PlantType
from .config import *

plant_counts = {
    PlantType.sunflower: 9,
    PlantType.pea_shooter: 6,
    PlantType.squash: 3,
    PlantType.snow_pea: 2,
}

zombie_deck = [
    [ZombieType.zombie, 50],
    [ZombieType.buckethead, 125],
    [ZombieType.football, 175],
]


class IZenv:
    def __init__(self, max_step=1_200, disable_col_6_to_9=False):
        self.action_no = N_ZOMBIE_TYPE * N_LANES * Z_LANE_LENGTH + 1
        self.reset()
        self._max_step = max_step
        self._step_count = 0
        self._brain_status = np.array([1, 1, 1, 1, 1])
        self._disable_col_6_to_9 = disable_col_6_to_9

    def reset(self):
        self.world = World(SceneType.night)
        self.world.scene.stop_spawn = True
        self.world.scene.is_iz = True
        self.world.scene.set_sun(150)
        plant_list = [
            plant for plant, count in plant_counts.items() for _ in range(count)
        ]
        np.random.shuffle(plant_list)
        for index, plant in enumerate(plant_list):
            x = index // P_LANE_LENGTH  # Row index
            y = index % P_LANE_LENGTH  # Column index
            self.world.plant_factory.create(plant, x, y)
        self._data = json.loads(self.world.get_json())
        self._step_count = 0
        self._brain_status = np.array([1, 1, 1, 1, 1])
        return self.get_state()

    def step(self, action):
        self._step_count += 1
        self._data = json.loads(self.world.get_json())
        prev_sun = self._get_sun()
        prev_brain_num = self._get_brain_num()

        self._take_action(action)
        for _ in range(50):
            self.world.update()

        self._data = json.loads(self.world.get_json())

        state = self.get_state()
        # print(state)
        reward = self._get_sun() - prev_sun
        if (self._has_lost(state)):
            reward = -500
        if (self._has_won(state)):
            reward += 500
        is_done = (
            self._has_lost(state)
            or self._has_won(state)
            or self._step_count >= self._max_step
        )
        return reward, state, is_done

    def _take_action(self, action):
        if action > 0:
            action -= 1
            z_idx = action // (N_LANES * Z_LANE_LENGTH)
            action_area = action % (N_LANES * Z_LANE_LENGTH)
            row = action_area // N_LANES
            col = action_area % N_LANES + 4
            sun = self._get_sun()
            # print(sun)
            sun -= zombie_deck[z_idx][1]
            if sun < 0:
                print("error, sun cannot be negative")
                return
            self.world.zombie_factory.create(zombie_deck[z_idx][0], row, col)
            # print(type(sun))
            # print(zombie_deck[z_idx][1])
            self.world.scene.set_sun(sun)

    def get_state(self):
        # Each plant HP, intialize to 0 to indicate if there is no plant
        plant_hps = np.zeros(N_LANES * P_LANE_LENGTH, dtype=float)
        # Each plant type, intialize to 0 to indicate if there is no plant
        plant_types = np.zeros(N_LANES * P_LANE_LENGTH, dtype=int)
        # Each zombie HP, intialize to 0 to indicate if there is no zombie
        zombie_hps = np.zeros(N_LANES * LANE_LENGTH, dtype=float)
        # Each zombie type, intialize to 0 to indicate if there is no zombie
        zombie_types = np.zeros(N_LANES * LANE_LENGTH, dtype=int)
        # Total sun
        sun = np.array([self._get_sun() / SUN_MAX])

        for plant in self._data["plants"]:
            plant_hps[plant["row"] * P_LANE_LENGTH + plant["col"]] = (
                plant["hp"] / P_MAX_HP
            )

            # Assign plant type depending on the plant type
            plant_type = 0
            if plant["type"] == "sunflower":
                plant_type = 1
            elif plant["type"] == "pea_shooter":
                plant_type = 2
            elif plant["type"] == "squash":
                plant_type = 3
            elif plant["type"] == "snow_pea":
                plant_type = 4

            plant_types[plant["row"] * P_LANE_LENGTH + plant["col"]] = plant_type / N_PLANT_TYPE

        for zombie in self._data["zombies"]:
            # Update brain status for each row
            if zombie["x"] < 25:
                self._brain_status[zombie["row"]] = 0

            # Calcullate zombie total HP and assign type depending on the zombie type
            zombie_hp = 0
            zombie_type = 0
            if zombie["type"] == "zombie":
                zombie_type = 1
                zombie_hp = zombie["hp"] * 0.66
            elif zombie["type"] == "buckethead":
                zombie_type = 2
                zombie_hp = zombie["hp"] * 0.66 + zombie["accessory_1"]["hp"]
            elif zombie["type"] == "football":
                zombie_type = 3
                zombie_hp = zombie["hp"] * 0.66 + zombie["accessory_1"]["hp"]

            zombie_hps[
                zombie["row"] * LANE_LENGTH + self._zombie_x_to_col(zombie["x"])
            ] += (zombie_hp / Z_MAX_HP)
            zombie_types[
                zombie["row"] * LANE_LENGTH + self._zombie_x_to_col(zombie["x"])
            ] = zombie_type / N_ZOMBIE_TYPE

        return np.concatenate(
            [
                plant_hps,
                plant_types,
                zombie_hps,
                zombie_types,
                sun,
                self._brain_status,
            ]
        )

    def get_valid_actions(self, state):
        # Example return:
        # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        # 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
        # 49 50] or
        # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        # 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
        # 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72
        # 73 74 75] or
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # 24 25]
        actions = np.arange(self.action_no)
        masks = self.get_action_mask(state)
        return actions[masks]

    def get_action_mask(self, state):
        sun = state[-6] * SUN_MAX
        z_no = self._get_zombie_num(state)
        mask = np.zeros(self.action_no, dtype=bool)
        if z_no > 0:
            mask[0] = True
        if sun >= 50:
            mask[1:26] = True
        if sun >= 125:
            mask[26:51] = True
        if sun >= 175:
            mask[51:] = True
        if self._disable_col_6_to_9:
            for i in range(1, 76):
                if i % 5 != 1:
                    mask[i] = False
        return mask

    def _zombie_x_to_col(self, x):
        col = (x - 10) / 80 - 1
        return math.ceil(col)

    def _get_sun(self):
        return self._data["sun"]["sun"]
    
    def _get_brain_num(self):
        return np.count_nonzero(self._brain_status == 0)

    def _get_reward(self):
        sun = self._get_sun()
        return sun // 25

    def _get_zombie_num(self, state):
        zombieNum = 45 + state[ZOMBIE_TYPE_START:ZOMBIE_TYPE_END].sum()

        return zombieNum

    def _has_lost(self, state):
        if self._has_won(state):
            return False
        
        sun = self._get_sun()
        zombieNum = self._get_zombie_num(state)

        if sun < 50 and zombieNum == 0:
            return True

        return False

    def _has_won(self, state):
        sum = state[-N_LANES:].sum()
        if sum == 0:
            return True

        return False
