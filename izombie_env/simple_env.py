import numpy as np
import ujson as json
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

BRAIN_VALUE = 100
LOSE_VALUE = -10 * BRAIN_VALUE
EAT_BRAIN_X_THRES = 25


class IZenv:
    def __init__(self, step_length=50, max_step=None, fix_rand=False):
        self.step_length = step_length
        self.max_step = max_step
        self.fix_rand = fix_rand

    def reset(self):
        self.step_count = 0
        self._brain_status = np.array([1, 1, 1, 1, 1])

        self.world = World(SceneType.night)
        self.world.scene.stop_spawn = True
        self.world.scene.is_iz = True
        self.world.scene.set_sun(150)
        plant_list = [
            plant for plant, count in plant_counts.items() for _ in range(count)
        ]
        if self.fix_rand:
            np.random.seed(0)
        np.random.shuffle(plant_list)
        for index, plant in enumerate(plant_list):
            self.world.plant_factory.create(
                plant, index // P_LANE_LENGTH, index % P_LANE_LENGTH
            )

        self._sync_with_world()
        return self.state

    def _get_game_status(self):
        if self.max_step is not None and self.step_count >= self.max_step:
            return GameStatus.TIMEUP
        if self._get_brain_num() == 0:
            return GameStatus.WIN
        if self.get_sun() < 50 and self._get_zombie_num() == 0:
            return GameStatus.LOSE
        return GameStatus.CONTINUE

    def _get_reward(self, prev_sun, prev_brain_num, game_status):
        reward = self.get_sun() - prev_sun
        reward += BRAIN_VALUE * (self._get_brain_num() - prev_brain_num)
        if game_status == GameStatus.LOSE:
            reward += LOSE_VALUE
        elif game_status == GameStatus.WIN:
            reward += self.get_sun()
        return reward

    def step(self, action):
        self.step_count += 1
        self._sync_with_world()
        prev_sun = self.get_sun()
        prev_brain_num = self._get_brain_num()

        self._take_action(action)
        for _ in range(self.step_length):
            self.world.update()
        self._sync_with_world()

        game_status = self._get_game_status()
        reward = self._get_reward(prev_sun, prev_brain_num, game_status)

        return reward, self.state, game_status

    def get_valid_actions(self):
        actions = np.arange(ACTION_NO)
        masks = self.get_action_mask()
        return actions[masks]

    def get_action_mask(self):
        sun = self.state[-6] * SUN_MAX
        mask = np.zeros(ACTION_NO, dtype=bool)
        if self._get_zombie_num() > 0:
            mask[0] = True
        if sun >= 50:
            mask[1:6] = True
        if sun >= 125:
            mask[6:11] = True
        if sun >= 175:
            mask[11:16] = True
        for row in range(5):
            if self._brain_status[row] == 0:
                for i in range(3):
                    mask[i * 5 + row + 1] = False
        return mask

    def print_human_readable_state(self):
        print(
            f"Step: {self.step_count} Sun: {self.get_sun()} Brains: {self._get_brain_num()} Game status: {self._get_game_status().name} "
        )
        state = self.state

        print("==Plant HP==")
        for row in range(5):
            print(
                f"row {row}: {state[row * P_LANE_LENGTH :(row+1)*P_LANE_LENGTH] * P_MAX_HP}"
            )

        state = state[5 * P_LANE_LENGTH :]
        print("==Plant Type==")
        for row in range(5):
            plant_types = (
                state[row * P_LANE_LENGTH : (row + 1) * P_LANE_LENGTH] * N_PLANT_TYPE
            )
            print(
                f"row {row}: {[self._plant_num_to_plant_type(p) for p in plant_types]}"
            )

        state = state[5 * P_LANE_LENGTH :]
        print("==Zombie HP==")
        for row in range(5):
            print(
                f"row {row}: {state[row * LANE_LENGTH:(row+1)*LANE_LENGTH] * Z_MAX_HP}"
            )

        state = state[5 * LANE_LENGTH :]
        print("==Zombie Type==")
        for row in range(5):
            zombie_types = (
                state[row * LANE_LENGTH : (row + 1) * LANE_LENGTH] * N_ZOMBIE_TYPE
            )
            print(
                f"row {row}: {[self._zombie_num_to_zombie_type(z) for z in zombie_types]}"
            )
        print()

    def _sync_with_world(self):
        self._data = json.loads(self.world.get_json())

        plant_hps = np.zeros(N_LANES * P_LANE_LENGTH, dtype=float)
        plant_type_nums = np.zeros(N_LANES * P_LANE_LENGTH, dtype=float)
        zombie_hps = np.zeros(N_LANES * LANE_LENGTH, dtype=float)
        zombie_type_nums = np.zeros(N_LANES * LANE_LENGTH, dtype=float)
        sun = np.array([self.get_sun() / SUN_MAX])

        for plant in self._data["plants"]:
            if plant["status"] == "squash_crushed":
                continue

            plant_hps[plant["row"] * P_LANE_LENGTH + plant["col"]] = (
                plant["hp"] / P_MAX_HP
            )
            plant_type_nums[plant["row"] * P_LANE_LENGTH + plant["col"]] = (
                self._plant_type_to_plant_num(plant["type"]) / N_PLANT_TYPE
            )

        for zombie in self._data["zombies"]:
            # Update brain status for each row
            if zombie["x"] < EAT_BRAIN_X_THRES:
                self._brain_status[zombie["row"]] = 0

            zombie_hp = zombie["hp"] * 0.66 + zombie["accessory_1"]["hp"]

            zombie_hps[
                zombie["row"] * LANE_LENGTH + self._zombie_x_to_col(zombie["x"])
            ] += (zombie_hp / Z_MAX_HP)
            zombie_type_nums[
                zombie["row"] * LANE_LENGTH + self._zombie_x_to_col(zombie["x"])
            ] = (self._zombie_type_to_zombie_num(zombie["type"]) / N_ZOMBIE_TYPE)

        self.state = np.concatenate(
            [
                plant_hps,
                plant_type_nums,
                zombie_hps,
                zombie_type_nums,
                sun,
                self._brain_status,
            ]
        )

    def _take_action(self, action):
        if action > 0:
            action -= 1
            z_idx = action // N_LANES
            row = action % N_LANES
            col = 4
            sun = self.get_sun() - zombie_deck[z_idx][1]
            assert sun >= 0
            self.world.zombie_factory.create(zombie_deck[z_idx][0], row, col)
            self.world.scene.set_sun(sun)

    def _zombie_x_to_col(self, x):
        col = int((x + 40) // 80)
        if col < 0:
            col = 0
        if col >= LANE_LENGTH:
            col = LANE_LENGTH - 1
        return col

    def get_sun(self):
        return self._data["sun"]["sun"]

    def _get_brain_num(self):
        return np.count_nonzero(self._brain_status == 1)

    def _get_zombie_num(self):
        return len(self._data["zombies"])

    def _plant_num_to_plant_type(self, plant_num):
        if plant_num == 1:
            return "sun"
        if plant_num == 2:
            return "pea"
        if plant_num == 3:
            return "sqa"
        if plant_num == 4:
            return "sno"
        return "___"

    def _plant_type_to_plant_num(self, plant_type):
        if plant_type == "sunflower":
            return 1
        if plant_type == "pea_shooter":
            return 2
        if plant_type == "squash":
            return 3
        if plant_type == "snow_pea":
            return 4
        return 0

    def _zombie_num_to_zombie_type(self, zombie_num):
        if zombie_num == 1:
            return "Z"
        if zombie_num == 2:
            return "B"
        if zombie_num == 3:
            return "F"
        return "_"

    def _zombie_type_to_zombie_num(self, zombie_type):
        if zombie_type == "zombie":
            return 1
        if zombie_type == "buckethead":
            return 2
        if zombie_type == "football":
            return 3
        return 0
